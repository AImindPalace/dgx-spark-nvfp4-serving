#!/bin/bash
# End-to-end DFlash test against the Cycle-2-merged fine-tuned target.
#
# 1. Download the DFlash drafter if missing.
# 2. Start the DFlash server on port 8001.
# 3. Wait for /health, snapshot /metrics.
# 4. Run shootout_harvest.py against it (20 prompts, 2048 tokens).
# 5. Snapshot /metrics again. Compute acceptance stats.
# 6. Stop the server. Print summary and point at the results JSON.
#
# A/B vs MTP: leave the MTP server running on port 8000 during this test.
# The two servers share the GPU — make sure gpu-memory-utilization is tuned
# so both fit (0.70 each is typical on 128 GB UMA; drop the MTP target to
# 0.35 if you hit OOM).
#
# Usage:
#   bash scripts/test_dflash.sh                      # default label
#   LABEL=dflash_v1 bash scripts/test_dflash.sh      # custom label

set -euo pipefail

LABEL="${LABEL:-dflash_jarvis27}"
PORT="${PORT:-8001}"
TARGET_PATH="${TARGET_PATH:-/home/brandonv/models/Jarvis_27B_trading}"
DRAFTER_PATH="${DRAFTER_PATH:-/home/brandonv/models/Qwen3.5-27B-DFlash}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
SERVER_LOG="${SERVER_LOG:-/tmp/dflash_server.log}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== DFlash A/B test ==="
echo "label         : $LABEL"
echo "target        : $TARGET_PATH"
echo "drafter       : $DRAFTER_PATH"
echo "port          : $PORT"
echo "max_tokens    : $MAX_TOKENS"
echo "server log    : $SERVER_LOG"
echo

# ---------------------------------------------------------------------------
# Step 1: ensure drafter is downloaded
# ---------------------------------------------------------------------------
if [[ ! -d "$DRAFTER_PATH" ]]; then
    echo "[1/6] Downloading drafter to $DRAFTER_PATH ..."
    hf download z-lab/Qwen3.5-27B-DFlash --local-dir "$DRAFTER_PATH"
else
    echo "[1/6] Drafter already present at $DRAFTER_PATH"
fi

# ---------------------------------------------------------------------------
# Step 2: start server
# ---------------------------------------------------------------------------
echo "[2/6] Starting DFlash server on port $PORT ..."
TARGET_PATH="$TARGET_PATH" DRAFTER_PATH="$DRAFTER_PATH" PORT="$PORT" \
    bash "$SCRIPT_DIR/serve_jarvis_27b_dflash.sh" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "      server PID: $SERVER_PID"

cleanup() {
    echo
    echo "[cleanup] Stopping server PID $SERVER_PID ..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Step 3: wait for /health
# ---------------------------------------------------------------------------
echo "[3/6] Waiting for /health (up to ${HEALTH_TIMEOUT_S}s) ..."
deadline=$(( $(date +%s) + HEALTH_TIMEOUT_S ))
until curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; do
    if [[ $(date +%s) -gt $deadline ]]; then
        echo "ERROR: server did not become healthy within ${HEALTH_TIMEOUT_S}s" >&2
        echo "tail of $SERVER_LOG:" >&2
        tail -50 "$SERVER_LOG" >&2
        exit 1
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server PID $SERVER_PID died before becoming healthy" >&2
        tail -80 "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 5
done
echo "      server is healthy"

# Snapshot metrics before the benchmark run
METRICS_BEFORE="/tmp/dflash_metrics_before.txt"
curl -fsS "http://localhost:$PORT/metrics" >"$METRICS_BEFORE" || true

# ---------------------------------------------------------------------------
# Step 4: run harvest
# ---------------------------------------------------------------------------
echo "[4/6] Running shootout_harvest.py against port $PORT ..."
python "$SCRIPT_DIR/shootout_harvest.py" \
    --label "$LABEL" \
    --base-url "http://localhost:$PORT" \
    --model "$TARGET_PATH" \
    --max-tokens "$MAX_TOKENS"

# ---------------------------------------------------------------------------
# Step 5: acceptance stats from /metrics
# ---------------------------------------------------------------------------
echo "[5/6] Collecting acceptance stats ..."
METRICS_AFTER="/tmp/dflash_metrics_after.txt"
curl -fsS "http://localhost:$PORT/metrics" >"$METRICS_AFTER"

python - <<'PY'
import re, os
before = open(os.environ.get("METRICS_BEFORE", "/tmp/dflash_metrics_before.txt")).read()
after = open(os.environ.get("METRICS_AFTER", "/tmp/dflash_metrics_after.txt")).read()

def scrape(text):
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        for key in ("vllm:spec_decode_num_accepted_tokens_total",
                    "vllm:spec_decode_num_draft_tokens_total",
                    "vllm:spec_decode_num_emitted_tokens_total",
                    "vllm:generation_tokens_total",
                    "vllm:prompt_tokens_total"):
            if line.startswith(key + " ") or line.startswith(key + "{"):
                m = re.search(r"\s([0-9.+eE-]+)\s*$", line)
                if m:
                    out[key] = out.get(key, 0.0) + float(m.group(1))
    return out

b = scrape(before)
a = scrape(after)
delta = {k: a.get(k, 0) - b.get(k, 0) for k in set(a) | set(b)}

accepted = delta.get("vllm:spec_decode_num_accepted_tokens_total", 0)
drafted = delta.get("vllm:spec_decode_num_draft_tokens_total", 0)
emitted = delta.get("vllm:spec_decode_num_emitted_tokens_total", 0)
gen     = delta.get("vllm:generation_tokens_total", 0)

print()
print(f"--- DFlash acceptance (during harvest run) ---")
print(f"generation_tokens_total (emitted to user): {gen:,.0f}")
print(f"spec_decode_num_accepted_tokens_total   : {accepted:,.0f}")
print(f"spec_decode_num_draft_tokens_total      : {drafted:,.0f}")
print(f"spec_decode_num_emitted_tokens_total    : {emitted:,.0f}")
if drafted > 0:
    print(f"per-token draft acceptance rate          : {accepted/drafted:.3f}")
if emitted > 0:
    print(f"mean acceptance length (accepted/emitted): {accepted/emitted:.3f}")
print()
PY

# ---------------------------------------------------------------------------
# Step 6: summary
# ---------------------------------------------------------------------------
echo "[6/6] Done. Results at:"
ls -1t "$REPO_ROOT/benchmarks/results" 2>/dev/null | grep -E "^$LABEL" | head -5 | sed "s|^|  $REPO_ROOT/benchmarks/results/|"
echo
echo "Next:"
echo "  1. Inspect completions and tok/s in the results JSON above."
echo "  2. Compare to MTP baseline: mean tok/s = 17.1 at 59 GB memory."
echo "  3. If you want blind quality scoring, run:"
echo "     python $SCRIPT_DIR/shootout_score.py --labels $LABEL jarvis_mtp_baseline"
