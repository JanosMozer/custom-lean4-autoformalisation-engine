"""Persistent Lean REPL pool for fast type-checking during RL.

Each worker is a long-lived `lake env repl` process (run over a pty via pexpect,
which forces Lean to flush per response) with Mathlib imported ONCE at startup.
Per-check cost then drops from ~20 s (cold `lake env lean`) to ~milliseconds.
Workers are handed out through a thread-safe queue so a ThreadPoolExecutor can
run many checks concurrently.
"""
import json
import logging
import os
import queue
import threading
from typing import Optional

import pexpect

logger = logging.getLogger("LeanServer")


class _Worker:
    def __init__(self, repl_bin: str, project: str, timeout: int):
        self.repl_bin, self.project, self.timeout = repl_bin, project, timeout
        self.c: Optional[pexpect.spawn] = None
        self.mathlib_env = 0
        self._start()

    def _start(self) -> None:
        self.c = pexpect.spawn(
            "lake", ["env", self.repl_bin], cwd=self.project,
            encoding="utf-8", echo=False, maxread=8192, timeout=self.timeout,
        )
        resp = self._cmd({"cmd": "import Mathlib"}, timeout=600)
        self.mathlib_env = (resp or {}).get("env", 0)

    def _cmd(self, obj: dict, timeout: int) -> Optional[dict]:
        """Send one command (JSON + blank line); accumulate output lines until it
        parses as a complete JSON reply. pty echo of the command is filtered out."""
        self.c.timeout = timeout
        self.c.sendline(json.dumps(obj))
        self.c.sendline("")
        buf = ""
        while True:
            line = self.c.readline()
            if line == "":
                raise pexpect.EOF("lean repl closed")
            s = line.strip()
            if not s or s.startswith('{"cmd"'):
                continue
            buf += line
            try:
                return json.loads(buf)
            except json.JSONDecodeError:
                continue

    def check(self, code: str) -> bool:
        return self.run(code)["ok"]

    def run(self, code: str) -> dict:
        """Full result: {ok, errors:[str], goal:str|None}. `goal` is the elaborated
        goal of the `sorry` (usable to compare two statements semantically)."""
        resp = self._cmd({"cmd": code, "env": self.mathlib_env}, self.timeout)
        if not resp:
            return {"ok": False, "errors": ["no repl response"], "goal": None}
        errors = [m.get("data", "") for m in resp.get("messages", []) if m.get("severity") == "error"]
        sorries = resp.get("sorries", [])
        goal = sorries[0].get("goal") if sorries else None
        return {"ok": len(errors) == 0, "errors": errors, "goal": goal}

    def restart(self) -> None:
        try:
            self.c.close(force=True)
        except Exception:
            pass
        self._start()


class LeanREPLPool:
    def __init__(self, repl_bin: str, project: str, size: int, timeout: int):
        self.size = size
        self._q: "queue.Queue[_Worker]" = queue.Queue()
        # Sequential spawn: concurrent `lake env` on the same project contend and
        # can hang. The first import warms the OS page cache so the rest are fast.
        logger.info(f"Spawning {size} Lean REPL workers (Mathlib preloaded once each)...")
        for i in range(size):
            self._q.put(_Worker(repl_bin, project, timeout))
            logger.info(f"  worker {i + 1}/{size} ready")
        logger.info("Lean REPL pool ready.")

    def check(self, code: str) -> bool:
        return self.run(code)["ok"]

    def run(self, code: str) -> dict:
        w = self._q.get()
        try:
            return w.run(code)
        except (pexpect.TIMEOUT, pexpect.EOF, OSError, ValueError):
            w.restart()  # recycle a dead/hung worker
            return {"ok": False, "errors": ["repl crashed"], "goal": None}
        finally:
            self._q.put(w)


def make_pool(repl_bin: str, project: str, size: int, timeout: int) -> "LeanREPLPool":
    """Persistent REPL pool with .check(code)->bool and .run(code)->dict."""
    repl_bin = os.path.abspath(repl_bin)
    project = os.path.abspath(project)
    if not os.path.exists(repl_bin):
        raise FileNotFoundError(f"repl binary not found at {repl_bin}")
    return LeanREPLPool(repl_bin, project, size, timeout)


def make_checker(repl_bin: str, project: str, size: int, timeout: int):
    """Return a `check(code)->bool`. Uses the persistent REPL pool if the repl
    binary exists, else falls back to cold `lake env lean` (slow)."""
    # Workers run with cwd=project, so the repl path must be absolute.
    repl_bin = os.path.abspath(repl_bin) if repl_bin else repl_bin
    project = os.path.abspath(project)

    if repl_bin and os.path.exists(repl_bin):
        return LeanREPLPool(repl_bin, project, size, timeout).check

    logger.warning(f"REPL binary not found at {repl_bin}; falling back to cold `lake env lean`.")
    import subprocess
    import tempfile

    def cold_check(code: str) -> bool:
        with tempfile.NamedTemporaryFile("w", suffix=".lean", dir=project, delete=False) as f:
            f.write(("" if "import Mathlib" in code else "import Mathlib\n") + code + "\n")
            path = f.name
        try:
            r = subprocess.run(["lake", "env", "lean", path], cwd=project,
                               capture_output=True, timeout=timeout)
            return r.returncode == 0
        except Exception:
            return False
        finally:
            os.unlink(path)

    return cold_check
