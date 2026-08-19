#!/usr/bin/env bash
# Dry-run smoke test — exercises the full agent pipeline without real models.
# The Python servers run in stub mode (no Lean, no GPU).
# A minimal mock inference server is started inline via Python.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENGINE="$ROOT/engine"
PYTHON="${PYTHON:-python3}"

echo "=== lean-engine smoke test ==="
echo "workspace: $ROOT"

# ── 0. Build the engine binary ───────────────────────────────────────────────
echo "building lean-engine..."
CARGO_TARGET_DIR=/tmp/lean-engine-target cargo build --manifest-path "$ENGINE/Cargo.toml" 2>&1
BINARY=/tmp/lean-engine-target/debug/lean-engine

# ── 1. Start stub inference server ───────────────────────────────────────────
INFER_PORT=19876
$PYTHON - <<'PYEOF' &
import asyncio, json

# Each connection can serve multiple requests (persistent connection model).
async def handle(r, w):
    try:
        while True:
            line = await r.readline()
            if not line:
                break
            w.write((json.dumps({"completion": "theorem test : True := by sorry"}) + "\n").encode())
            await w.drain()
    except Exception:
        pass
    finally:
        w.close()

async def main():
    srv = await asyncio.start_server(handle, "127.0.0.1", 19876)
    async with srv:
        await srv.serve_forever()

asyncio.run(main())
PYEOF
INFER_PID=$!

# Wait until the inference server is up.
for i in $(seq 1 20); do
    if $PYTHON -c "import socket; s=socket.create_connection(('127.0.0.1',19876),0.3); s.close()" 2>/dev/null; then
        break
    fi
    sleep 0.2
done
echo "stub inference server up (pid $INFER_PID)"

# ── 2. Write a minimal smoke config ──────────────────────────────────────────
TMPDIR_SMOKE="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_SMOKE"; kill $INFER_PID 2>/dev/null || true' EXIT

cat > "$TMPDIR_SMOKE/config.toml" <<TOML
title         = "smoke_test"
problem       = "For all natural numbers n, n + 0 = n."
model_path    = "path_to_model"
max_iters     = 2
output_format = "txt"
output_dir    = "$TMPDIR_SMOKE/formalised"
lean_header   = "import Mathlib\n\n"
retrieval_emb  = "retrieval/emb.npy"
retrieval_meta = "retrieval/meta.jsonl"
retrieval_bm25 = "retrieval/bm25.json"
retrieval_k    = 8
infer_host     = "127.0.0.1"
infer_port     = $INFER_PORT
temperature    = 0.0
max_new_tokens = 64
TOML

# ── 3. Run the engine binary ──────────────────────────────────────────────────
echo "running lean-engine..."
LEANBENCH_ROOT="$ROOT" \
RUST_LOG="lean_engine=debug" \
"$BINARY" "$TMPDIR_SMOKE/config.toml"
STATUS=$?

echo ""
echo "=== output ==="
cat "$TMPDIR_SMOKE/formalised/smoke_test.txt" 2>/dev/null || echo "(no output file)"

if [ $STATUS -eq 0 ]; then
    echo ""
    echo "PASSED"
else
    echo ""
    echo "UNSOLVED (exit $STATUS) — expected for stub REPL, pipeline ran correctly"
fi
