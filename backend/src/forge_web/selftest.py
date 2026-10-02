"""``forge_web`` 自检。

覆盖那些**不会报错、只会让网页默默失灵**的地方：
路径穿越、SSE 事件漏发、领域映射越界、并发上限、子进程回收。
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from .app import create_app
from .jobs import ArchSpec, TrainSpec
from .settings import Settings


def _app(tmp: Path) -> TestClient:
    st = Settings(project_root=tmp, max_concurrent_runs=1)
    return TestClient(create_app(st))


def test_health_has_no_torch(m: object = None) -> None:
    """/api/health 不能把 torch 拖进 API 进程。

    API 进程一旦 import 过 torch 并初始化 CUDA context，就会占住一块
    显存不放，训练进程再分配时就更容易 OOM——而且很难查。
    """
    import sys as _s
    with tempfile.TemporaryDirectory() as td:
        c = _app(Path(td))
        r = c.get("/api/health")
        assert r.status_code == 200, r.text
        assert "torch" not in _s.modules or _s.modules["torch"] is None or \
            not getattr(_s.modules["torch"], "cuda", None) or \
            not _s.modules["torch"].cuda.is_initialized(), \
            "API 进程初始化了 CUDA context"


def test_domains_endpoint() -> None:
    with tempfile.TemporaryDirectory() as td:
        c = _app(Path(td))
        d = c.get("/api/domains").json()
        assert len(d["domains"]) == 32, len(d["domains"])
        assert len(d["default_map"]) == 32
        # 默认映射必须让每个大核都有领域，否则大核会一上来就死
        from collections import Counter
        cnt = Counter(d["default_map"].values())
        assert len(cnt) == 4, f"默认映射只用了 {len(cnt)} 个大核"
        assert min(cnt.values()) == 8, cnt


def test_path_traversal_blocked() -> None:
    """``data/../../etc/passwd`` 必须被挡在允许目录外。"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "data").mkdir()
        c = _app(tmp)
        r = c.post("/api/runs", json={
            "architecture": {"name": "x"},
            "dataset_path": "data/../../../etc/passwd", "max_steps": 1})
        assert r.status_code in (403, 400), (r.status_code, r.text[:200])


def test_arch_spec_rejects_out_of_range() -> None:
    """越界必须按**当前**的大核数判定，不能用硬编码上界。"""
    try:
        ArchSpec(name="bad", num_macro_cores=2,
                 domain_to_macro={"Arts_Poetry": 7})
    except Exception as e:                       # noqa: BLE001
        assert "超出" in str(e), e
    else:
        raise AssertionError("越界的领域映射竟然被接受了")


def test_arch_spec_rejects_unknown_domain() -> None:
    try:
        ArchSpec(name="bad", domain_to_macro={"Not_A_Domain": 0})
    except Exception as e:                       # noqa: BLE001
        assert "未知领域" in str(e), e
    else:
        raise AssertionError("未知领域竟然被接受了")


def test_arch_spec_fills_missing_domains_evenly() -> None:
    """只拖动部分领域时，剩下的必须**均衡**补齐。

    不能默认全填 0——那会让大核 1..M-1 一个领域都没有，直接死掉。
    """
    a = ArchSpec(name="x", num_macro_cores=4,
                 domain_to_macro={"Arts_Poetry": 3})
    full = a.resolved_map()
    from collections import Counter
    cnt = Counter(full.values())
    assert set(cnt) == {0, 1, 2, 3}, f"有的大核没领域：{cnt}"
    assert full["Arts_Poetry"] == 3
    assert min(cnt.values()) >= 7, cnt


def test_arch_names_length_mismatch() -> None:
    try:
        ArchSpec(name="x", num_macro_cores=3,
                 macro_names=["A", "B"]).resolved_names()
    except ValueError as e:
        assert "macro_names" in str(e), e
    else:
        raise AssertionError("macro_names 数量不符竟然被接受了")


def test_max_macro_cores_one() -> None:
    """M=1 是合法退化情形（第 0 号就是底座本身）。"""
    a = ArchSpec(name="x", num_macro_cores=1)
    assert a.to_profile().num_macro_cores == 1


def test_dataset_endpoint_generates_and_audits() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        c = _app(tmp)
        r = c.post("/api/datasets", json={
            "name": "tiny", "per_domain": 4, "domains": ["Arts_Poetry"],
            "out_name": "tiny.jsonl"})
        assert r.status_code == 200, r.text[:400]
        body = r.json()
        assert body["rows"] == 4, body
        # 内容唯一率必须是真实算出来的，不是 builder 的字节级 100%
        assert 0 < body["content_unique_ratio"] <= 1.0

        listed = c.get("/api/datasets").json()
        assert len(listed) == 1
        pv = c.get(f"/api/datasets/{listed[0]['id']}/preview").json()
        assert len(pv["rows"]) == 3


def test_run_rejects_missing_dataset() -> None:
    with tempfile.TemporaryDirectory() as td:
        c = _app(Path(td))
        r = c.post("/api/runs", json={
            "architecture": {"name": "x"},
            "dataset_path": "data/nope.jsonl", "max_steps": 1})
        assert r.status_code == 400, (r.status_code, r.text[:200])


def test_run_lifecycle_and_sse() -> None:
    """真跑一个 2 步的训练，验证状态机与 SSE 都能跑通。

    刻意用 ``forge_core.cli smoke`` 的等价路径（真子进程、真 NDJSON），
    因为这里要验的是"进程隔离 + 事件转发"，不是训练本身。
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        c = _app(tmp)

        # 造一份极小数据集（仓库根目录的 data/）
        src = (Path(__file__).resolve().parents[3]
               / "data" / "dual_contrast_data.jsonl")
        if not src.exists():
            print("     (跳过：数据集不存在)")
            return
        (tmp / "data").mkdir(exist_ok=True)
        small = tmp / "data" / "small.jsonl"
        small.write_text("".join(src.read_text(encoding="utf-8").splitlines(
            keepends=True)[:40]), encoding="utf-8")

        r = c.post("/api/runs", json={
            "architecture": {"name": "mini", "num_macro_cores": 2,
                             "num_micro_experts": 4, "macro_rank": 4,
                             "micro_rank": 2},
            "dataset_path": "data/small.jsonl",
            "base_model_id": "dummy/base",
            "max_steps": 1, "batch_size": 1, "grad_accum": 1,
            "max_length": 64, "device": "cpu", "param_dtype": "fp32",
            "text_only": True})
        # dummy/base 拉不到模型是预期的；我们要验的是
        # "run 被登记 + 状态能走到 failed 且 SSE 收得到 state 事件"
        assert r.status_code == 200, r.text[:400]
        run_id = r.json()["id"]

        events = []
        with c.stream("GET", f"/api/runs/{run_id}/events") as resp:
            assert resp.status_code == 200
            deadline = time.time() + 90
            for line in resp.iter_lines():
                if time.time() > deadline:
                    raise AssertionError("SSE 90s 内没有收到终态事件")
                if line.startswith("data:"):
                    events.append(json.loads(line[5:].strip()))
                if events and events[-1].get("t") == "state":
                    break

        assert events, "没有收到任何事件"
        assert events[-1]["t"] == "state"
        assert events[-1]["state"] in {"succeeded", "failed", "cancelled"}
        # seq 必须严格递增，否则前端重连补发会乱序
        seqs = [e["seq"] for e in events]
        assert seqs == sorted(seqs), seqs


def test_concurrency_limit() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        app = create_app(Settings(project_root=tmp, max_concurrent_runs=1))
        sup = app.state.supervisor
        c = TestClient(app)
        # 直接对 supervisor 塞一个长跑的进程来占住名额
        h = sup.spawn(999, ["-c", "import time; time.sleep(30)"], cwd=tmp)
        try:
            with pytest_raises(RuntimeError):
                sup.spawn(998, ["-c", "pass"], cwd=tmp)
        finally:
            h.stop(timeout=5)


def pytest_raises(exc):
    class _Ctx:
        def __enter__(self): return self
        def __exit__(self, et, ev, tb):
            assert et is not None, "期望抛异常但没有"
            assert issubclass(et, exc), f"期望 {exc}，实际 {et}"
            return True
    return _Ctx()


def test_path_allow_list() -> None:
    st = Settings(project_root=Path("/tmp"), allow_paths=(Path("/etc"),))
    assert "/etc" in str(st.resolved_allow())


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:                      # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
