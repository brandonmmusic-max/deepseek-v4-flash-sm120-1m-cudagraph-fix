#!/usr/bin/env bash
# =============================================================================
# Estonia long-context reasoning test (live TUI) vs the local DeepSeek-V4-Flash server.
# Double-click "Estonia Test" on the desktop, or run this directly.
# Correct answer = "Estonia". A correct server (native indexer + reasoning + greedy) = 30/30.
# =============================================================================
set -u
PORT="${1:-9201}"
HOST="${HOST:-127.0.0.1}"
BASE="$HOME/.local/share/estonia-tui"
VENV="$BASE/venv"
BENCH="$BASE/llm-inference-bench/llm_decode_bench.py"
mkdir -p "$BASE"

# ---- one-time setup (venv + benchmark) -------------------------------------
if [ ! -x "$VENV/bin/python3" ]; then
  echo "[setup] creating python venv (one time)..."
  python3 -m venv "$VENV" && "$VENV/bin/pip" -q install --upgrade pip >/dev/null && \
  "$VENV/bin/pip" -q install httpx rich psutil >/dev/null
fi
if [ ! -f "$BENCH" ]; then
  echo "[setup] fetching llm-inference-bench (one time)..."
  git -C "$BASE" clone --depth 1 https://github.com/local-inference-lab/llm-inference-bench >/dev/null 2>&1
fi

# ---- preflight: is the server up? ------------------------------------------
if [ "$(curl -s -m4 -o /dev/null -w '%{http_code}' "http://$HOST:$PORT/v1/models" 2>/dev/null)" != "200" ]; then
  echo "!! No V4-Flash server on $HOST:$PORT . Launch it first (launch_v4flash_1m_tp2.sh), then re-run."
  read -rp "Press Enter to close..."; exit 1
fi

echo "==============================================================="
echo " estonia 134K-token multi-hop reasoning test  (GREEDY, live TUI)"
echo " server: http://$HOST:$PORT   answer must be: Estonia"
echo "==============================================================="
"$VENV/bin/python3" "$BENCH" --host "$HOST" --port "$PORT" --model deepseek-v4-flash \
  --test-profile estonia --completion-stats-temperature 0 --display-mode live \
  --output "$BASE/last_estonia_result.json"
echo
echo "Result JSON saved to: $BASE/last_estonia_result.json"
read -rp "Done. Press Enter to close..."
