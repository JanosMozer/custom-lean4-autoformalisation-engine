#!/usr/bin/env python3
"""
Persistent REPL server — reads JSON-line requests from stdin, writes JSON-line
responses to stdout. Imports Mathlib once; subsequent checks cost only the REPL
eval time (~ms).

Protocol:
  request:  {"code": "<lean source>"}
  response: {"ok": bool, "errors": [...], "goal": str|null}
  shutdown: {"shutdown": true}  -> exits cleanly
"""
import json
import logging
import sys
import os

logging.basicConfig(
    stream=sys.stderr,
    format="%(asctime)s [repl_server] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("repl_server")

sys.path.insert(0, os.environ.get("LEANBENCH_ROOT", "."))

try:
    from src.repl import LeanREPLPool
    pool = LeanREPLPool(int(os.environ.get("REPL_WORKERS", "1")))
    log.info("LeanREPLPool ready (%s workers)", os.environ.get("REPL_WORKERS", "1"))
except Exception as exc:
    log.warning("LeanREPLPool unavailable (%s) — running in stub mode", exc)
    pool = None


def _stub_run(code: str):
    class R:
        ok = False
        errors = ["[stub] REPL not available"]
        goal = None
    return R()


def handle(req: dict) -> dict:
    code = req.get("code", "")
    r = pool.run(code) if pool else _stub_run(code)
    return {"ok": r.ok, "errors": list(r.errors), "goal": r.goal}


def main():
    log.info("ready — waiting for requests on stdin")
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError as e:
            sys.stdout.write(json.dumps({"ok": False, "errors": [f"bad JSON: {e}"], "goal": None}) + "\n")
            sys.stdout.flush()
            continue

        if req.get("shutdown"):
            log.info("shutdown requested")
            break

        try:
            resp = handle(req)
        except Exception as exc:
            log.exception("error handling request")
            resp = {"ok": False, "errors": [str(exc)], "goal": None}

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()

    log.info("exiting")


if __name__ == "__main__":
    main()
