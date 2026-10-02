"""训练子进程监管器。

为什么必须开子进程，而不是在 FastAPI 进程里直接 ``train()``
----------------------------------------------------------
1. **显存不可共享**。训练进程会吃掉十几 GB 显存且不释放；
   API 进程如果也 import 了 torch 并碰过 CUDA allocator，
   两者就会互相抢显存，一个 OOM 连带把网站也搞挂。
   隔离之后 API 进程**完全不 import torch**。
2. **能真正杀掉**。训练中崩了/卡了/用户想停，
   ``Popen.terminate()`` 能整组收掉；在同一进程里你只能等它自己结束。
3. **崩溃不扩散**。一段 CUDA 越界能直接杀死整个解释器，
   连带网站一起消失。
4. **不用重载模型**。换一次训练配置就换子进程，
   不用在 API 进程里反复 ``from_pretrained``。

事件流
------
``forge_core.cli`` 全程输出 NDJSON 到 stdout。监管器逐行读、逐条转成
SSE 事件。刻意**不解析** stderr 里的进度条——那格式会变，
解析它只会引入一个永远会坏的依赖；stderr 只在出错时整段回传。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterator

from .forge_core_bridge import backend_src, python_exe, sanitize_env


class RunHandle:
    """一个训练子进程 + 它的内存事件缓冲。"""

    def __init__(self, run_id: int, proc: subprocess.Popen,
                 log_path: Path) -> None:
        self.run_id = run_id
        self.proc = proc
        self.log_path = log_path
        self.started = time.time()
        self.state = "running"
        self.error: str | None = None

        self._lock = threading.Lock()
        self._seq = 0
        #: 只保留最近若干条，供"晚加入"的 SSE 订阅者补历史
        self._backlog: deque[dict[str, Any]] = deque(maxlen=500)
        #: 每个等待者各有一条 Condition
        self._cond = threading.Condition(self._lock)
        self.closed = False

    # -- 事件分发 -----------------------------------------------------
    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._seq += 1
            event["seq"] = self._seq
            self._backlog.append(event)
            self._cond.notify_all()

    def close(self, state: str, error: str | None = None) -> None:
        with self._lock:
            self.state = state
            self.error = error
            self.closed = True
            self._cond.notify_all()

    def since(self, seq: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self._backlog if e["seq"] > seq]

    def poll(self, after: int) -> tuple[list[dict[str, Any]], bool,
                                        str, str | None]:
        """**非阻塞**取一批新事件。

        给 asyncio 用。不能用 :meth:`stream`——它内部在
        ``Condition.wait_for`` 上阻塞，直接在事件循环里迭代会卡死整个
        服务（表现为"训练还在跑，但网页所有接口都没响应"）。
        """
        with self._lock:
            batch = [e for e in self._backlog if e["seq"] > after]
            return batch, self.closed, self.state, self.error

    def stream(self, after: int = 0, poll: float = 0.25) -> Iterator[dict[str, Any]]:
        """阻塞式事件迭代器，供后台**线程**使用（如 :func:`_persist_events`）。"""
        cursor = after
        while True:
            with self._cond:
                self._cond.wait_for(
                    lambda: self._seq > cursor or self.closed, timeout=poll)
                batch = [e for e in self._backlog if e["seq"] > cursor]
                closed, state, error = self.closed, self.state, self.error
            for e in batch:
                cursor = e["seq"]
                yield e
            if closed:
                if cursor < self._seq:
                    continue
                yield {"t": "state", "state": state, "error": error,
                       "seq": cursor + 1}
                return

    # -- 控制 ---------------------------------------------------------
    def stop(self, timeout: float = 10.0) -> str:
        """终止整个进程组。

        必须杀**进程组**：``subprocess`` 起的训练可能再 fork 出 dataloader
        worker，只 ``kill()`` 父进程会留下孤儿继续占显存。
        """
        if self.proc.poll() is not None:
            return "already_exited"
        try:
            pgid = os.getpgid(self.proc.pid)
        except ProcessLookupError:
            pgid = None

        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                self.proc.terminate()
        except ProcessLookupError:
            return "already_exited"

        try:
            self.proc.wait(timeout=timeout)
            return "terminated"
        except subprocess.TimeoutExpired:
            if pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                self.proc.kill()
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                return "kill_failed"
            return "killed"


class Supervisor:
    """全局单例：持有当前所有活跃 run。"""

    def __init__(self, *, log_dir: Path, max_concurrent: int = 1) -> None:
        self.log_dir = log_dir
        self.max_concurrent = max_concurrent
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._runs: dict[int, RunHandle] = {}
        self._lock = threading.Lock()

    def active(self) -> list[int]:
        with self._lock:
            return [rid for rid, h in self._runs.items()
                    if h.proc.poll() is None]

    def get(self, run_id: int) -> RunHandle | None:
        with self._lock:
            return self._runs.get(run_id)

    def spawn(self, run_id: int, argv: list[str], *,
              cwd: Path, env_extra: dict[str, str] | None = None
              ) -> RunHandle:
        with self._lock:
            n_active = sum(1 for h in self._runs.values()
                           if h.proc.poll() is None)
            if n_active >= self.max_concurrent:
                raise RuntimeError(
                    f"已有 {n_active} 个训练在跑（上限 "
                    f"{self.max_concurrent}）。GPU 只有一块，"
                    "并发训练必然互相抢显存导致 OOM。")

        env = os.environ.copy()
        sanitize_env()                       # 改的是 os.environ
        env.update(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(backend_src()), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        # 子进程要能写 HF 缓存但不需要联网；代理已由 sanitize_env 清掉。
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("HF_HUB_OFFLINE", "1")
        env["TOKENIZERS_PARALLELISM"] = "false"
        if env_extra:
            env.update(env_extra)

        log_path = self.log_dir / f"run-{run_id}.log"
        log_f = open(log_path, "wb")

        # start_new_session=True -> 独立的进程组，
        # 这样 stop() 能用 killpg 收掉整棵树。
        proc = subprocess.Popen(
            [python_exe(), "-u", *argv],
            cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True, bufsize=1,
        )
        handle = RunHandle(run_id, proc, log_path)
        with self._lock:
            self._runs[run_id] = handle
        threading.Thread(target=self._pump, args=(handle, log_f),
                         daemon=True).start()
        return handle

    def _pump(self, handle: RunHandle, log_f) -> None:
        """读子进程 stdout，逐行转事件。"""
        assert handle.proc.stdout is not None
        try:
            for raw in handle.proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                try:
                    log_f.write((line + "\n").encode("utf-8"))
                except ValueError:             # 日志文件被关掉了
                    pass
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # 非 NDJSON（进度条、警告）——原样透传，
                    # 前端决定要不要显示。不要试图"猜"它的格式。
                    handle.publish({"t": "log", "line": line})
                    continue
                handle.publish(event)
        except Exception as e:                  # noqa: BLE001
            handle.close("failed", f"读取子进程输出失败：{e}")
        finally:
            try:
                log_f.close()
            except Exception:                   # noqa: BLE001
                pass
            rc = handle.proc.wait()
            if rc == 0:
                handle.close("succeeded")
            elif rc < 0:
                handle.close("cancelled", f"被信号 {-rc} 终止")
            else:
                handle.close("failed", f"退出码 {rc}")


__all__ = ["Supervisor", "RunHandle"]
