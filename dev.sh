#!/usr/bin/env bash
# 前后端一键起停。
#
#   ./dev.sh            启动（等同 start）
#   ./dev.sh stop       停止（detached 启动的也能停）
#   ./dev.sh restart    重启
#   ./dev.sh status     查看状态
#
# 环境地雷：本机设置了 ALL_PROXY=socks://127.0.0.1:10808，
# huggingface_hub 解析不了 socks:// scheme，任何模型加载都会抛
# "Unknown scheme for proxy URL"。后端启动时会自动清掉它
# （forge_web.forge_core_bridge.sanitize_env），这里也显式清一次，
# 保证 CLI 直接跑时同样干净。
set -euo pipefail
cd "$(dirname "$0")"

RUN_DIR=".run"
BACKEND_PORT=8848
FRONTEND_PORT=5173
mkdir -p "$RUN_DIR" logs

PY=.venv/bin/python
[[ -x $PY ]] || PY=python3

# ---------------------------------------------------------------------------
# 关键：每个服务用 setsid 单独开一个**进程组**
# ---------------------------------------------------------------------------
# 为什么不用 `kill 0`：
#   `kill 0` 杀的是**当前进程组**，而 dev.sh 通常和用户的终端 shell
#   在同一组里 —— 那一刀会把用户的终端一起带走。
#
# 为什么不用普通 `&`：
#   uvicorn --reload 会再 fork 一个 worker，vite 会再 fork 一个
#   esbuild service。只杀父进程，这些孤儿会继续占着 8848/5173 端口，
#   下次启动就报 "Address already in use"。
#
# setsid 让每个服务自成组，记录组长 PID 后 `kill -- -PID` 就能整棵收掉。
spawn() {                       # spawn <name> <cmd...>
  local name=$1; shift
  setsid "$@" > "logs/$name.log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$RUN_DIR/$name.pid"
  printf '  %-8s pid %-7s 日志 logs/%s.log\n' "$name" "$pid" "$name"
}

is_running() {                 # is_running <name>
  local f=$RUN_DIR/$1.pid
  [[ -f $f ]] || return 1
  local pid; pid=$(cat "$f")
  [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null
}

stop_one() {                    # stop_one <name>
  local name=$1 f=$RUN_DIR/$1.pid
  if ! is_running "$name"; then
    rm -f "$f"
    return 1
  fi
  local pid; pid=$(cat "$f")
  # 先礼后兵：给 SIGTERM 的机会（vite/uvicorn 要时间收尾），超时再 SIGKILL
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.2
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    sleep 0.3
  fi
  rm -f "$f"
  return 0
}

# 端口上还挂着、但 PID 文件已经没了的孤儿进程
reap_by_port() {
  local port=$1 label=$2 pid cwd
  for pid in $(ss -ltnpH "sport = :$port" 2>/dev/null \
               | grep -oP 'pid=\K[0-9]+' | sort -u); do
    # 归属判断必须看**工作目录**，不能 grep 命令行：
    # 用相对路径（.venv/bin/python）启动时，cmdline 里根本不含仓库
    # 绝对路径，grep $PWD 会漏判，孤儿就永远收不掉。
    # /proc/<pid>/cwd 是内核维护的真实路径，不受 argv[0] 影响。
    cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
    case "$cwd" in
      "$PWD"|"$PWD"/*) ;;
      *) continue ;;   # 工作目录不在本项目，绝不误杀
    esac
    echo "  回收孤儿 $label (pid $pid, cwd=${cwd#"$PWD"/})"
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 15); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.2
    done
    kill -KILL "$pid" 2>/dev/null || true
    sleep 0.2
  done
  return 0
}

do_stop() {
  local any=0
  for s in backend frontend; do
    if stop_one "$s"; then echo "  已停止 $s"; any=1; fi
  done
  reap_by_port "$BACKEND_PORT" backend
  reap_by_port "$FRONTEND_PORT" frontend

  # 训练子进程是 API 用 start_new_session 起的，**不在**上面这些组里，
  # 所以停服务不会连带停训练。这里只提醒，不擅自 kill —— 训练可能是
  # 跑了半小时的成果。
  local runs
  runs=$(pgrep -f "forge_core.cli train" 2>/dev/null || true)
  if [[ -n $runs ]]; then
    echo
    echo "  ⚠ 仍有训练在跑（pid: $(echo "$runs" | tr '\n' ' ')）"
    echo "    停服务不会停训练。要停就执行："
    echo "    kill $runs"
  fi

  [[ $any -eq 0 ]] && echo "  本来就没在运行"
  return 0
}

do_start() {
  if is_running backend || is_running frontend; then
    echo "已在运行。要重来请先 ./dev.sh stop"; echo
    do_status; return 1
  fi

  echo "启动中…"
  spawn backend env -u ALL_PROXY -u all_proxy PYTHONPATH=backend/src \
    "$PY" -m uvicorn forge_web.app:app \
    --host 127.0.0.1 --port "$BACKEND_PORT" --reload

  (cd frontend && pnpm install --silent)
  spawn frontend pnpm --dir frontend dev --port "$FRONTEND_PORT" --strictPort

  # 等后端真的能应答，免得用户一打开页面就看到"无法连接"
  for _ in $(seq 1 40); do
    if curl -sf --noproxy '*' "http://127.0.0.1:$BACKEND_PORT/api/health" >/dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done

  echo
  echo "  后端  http://127.0.0.1:$BACKEND_PORT/api/health"
  echo "  前端  http://127.0.0.1:$FRONTEND_PORT"
  echo
  echo "  停止：./dev.sh stop"
  echo
}

do_status() {
  for s in backend frontend; do
    if is_running "$s"; then
      printf '  %-8s 运行中 (pid %s)\n' "$s" "$(cat "$RUN_DIR/$s.pid")"
    else
      printf '  %-8s 未运行\n' "$s"
    fi
  done
  # 只有"服务已停、端口却还亮着"才是异常——那说明有 PID 文件之外的
  # 孤儿进程占着端口，下次 start 会直接 "Address already in use"。
  # 服务自己在跑时端口当然被占用，那不是异常。
  check_port() {
    local name=$1 port=$2
    ss -ltnH "sport = :$port" 2>/dev/null | grep -q . || return 0
    if is_running "$name"; then return 0; fi
    echo "  ⚠ 端口 $port 仍被占用但 $name 未运行（孤儿进程？）"
  }
  check_port backend "$BACKEND_PORT"
  check_port frontend "$FRONTEND_PORT"
  return 0
}

case "${1:-start}" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_stop; echo; do_start ;;
  status)  do_status ;;
  *)       echo "用法: $0 [start|stop|restart|status]"; exit 2 ;;
esac
