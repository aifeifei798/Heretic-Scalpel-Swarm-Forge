#!/usr/bin/env bash
# 同时起后端(8848) 与前端(5173)，Ctrl-C 一起停。
#
# 注意：本机设置了 ALL_PROXY=socks://...，huggingface_hub 解析不了 socks://
# scheme，任何模型加载都会抛 "Unknown scheme for proxy URL"。
# 后端启动时会自动清掉它（forge_web.forge_core_bridge.sanitize_env），
# 但这里也显式清一次，保证 CLI 直接跑时同样干净。
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
[[ -x $PY ]] || PY=python3

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

env -u ALL_PROXY -u all_proxy PYTHONPATH=backend/src \
  $PY -m uvicorn forge_web.app:app --host 127.0.0.1 --port 8848 --reload &

sleep 2
(cd frontend && pnpm install --silent && pnpm dev) &

echo
echo "  后端  http://127.0.0.1:8848/api/health"
echo "  前端  http://127.0.0.1:5173"
echo
wait
