# engine/

Rust orchestrator for high-throughput Lean 4 autoformalization using the A+B retrieval system and an agentic REPL compiler-feedback loop.

## Structure

```
engine/
├── config.toml          # problem, model path, iteration budget, output format
├── src/
│   ├── main.rs          # entry point
│   ├── config.rs        # config loader
│   ├── model.rs         # async TCP client → Python inference server
│   ├── agent.rs         # A+B agentic loop (up to max_iters repair turns)
│   ├── repl.rs          # LeanREPLPool bridge + ident extractor
│   ├── retrieval.rs     # System A (lookup) + System B (hybrid retrieve)
│   └── output.rs        # txt / yaml / json result writer
├── scripts/
│   └── infer_server.py  # Python TCP server — loads QLoRA model, streams completions
└── formalised/          # output directory (auto-created)
```

## Prerequisites

- Rust ≥ 1.78
- Python ≥ 3.10 with `.venv` at workspace root (`pip install -r requirements.txt`)
- Lean 4 + Mathlib (lake env repl)
- GPU with ≥ 16 GB VRAM (NF4 model)

## Usage

**1. Start the inference server** (once per session, external process):

```bash
MODEL_PATH=/path/to/qlora_checkpoint python engine/scripts/infer_server.py
```

**2. Edit `engine/config.toml`** — set `title`, `problem`, `model_path`, `max_iters`, `output_format`.

**3. Run the engine** from the workspace root:

```bash
CARGO_TARGET_DIR=/tmp/lean-engine-target \
cargo run --release --manifest-path engine/Cargo.toml -- engine/config.toml
```

Output is written to `engine/formalised/<TITLE>.<ext>`.

**Smoke test** (no models required):

```bash
bash engine/scripts/smoke_test.sh
```

## Environment

| Variable | Default | Description |
|---|---|---|
| `LEANBENCH_ROOT` | `.` | Workspace root (for retrieval + REPL paths) |
| `MODEL_PATH` | `path_to_model` | Override model path for the server |
| `INFER_HOST` / `INFER_PORT` | `127.0.0.1:9876` | Inference server socket |
| `RUST_LOG` | `lean_engine=info` | Tracing filter |

## Logging

Set `RUST_LOG=lean_engine=debug` for verbose per-iteration traces.
