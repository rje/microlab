"""Sandboxed execution of untrusted Python programs — the backbone of the code evals and,
later, of execution-verified GRPO rewards.

Every run gets:

- a fresh subprocess in its own session/process group (``start_new_session=True``), so a
  wall-clock timeout SIGKILLs the whole group, including anything the program forked;
- a hard address-space cap via ``resource.setrlimit(RLIMIT_AS)`` — allocations beyond it
  fail, which surfaces as MemoryError/abort in the child, never pressure on the host;
- an ``RLIMIT_CPU`` backstop one notch above the wall-clock timeout (catches a program
  that blocks our pipe-draining but keeps burning CPU);
- an ``RLIMIT_FSIZE`` cap: stdout/stderr are redirected to files inside the sandbox
  tmpdir, so a print-bomb hits the file-size limit (SIGXFSZ) instead of our memory;
- a throwaway temporary directory as cwd/HOME/TMPDIR, deleted afterwards;
- a from-scratch whitelist environment (PATH/HOME/TMPDIR/LC_ALL only) — no proxy vars, no
  credentials, no PYTHONPATH (and ``python -I`` ignores user site-packages regardless);
- network isolation via ``unshare -rn`` (a private user+net namespace with only a dead
  loopback) WHEN the kernel allows unprivileged user namespaces. This is probed once per
  process; where it is unavailable (e.g. Ubuntu 24.04's default
  ``apparmor_restrict_unprivileged_userns=1``) the clean environment still strips proxies
  and credentials, but sockets are NOT blocked. The result records which level ran
  (``network_isolation``: "netns" or "env-only") and callers that require the hard
  guarantee pass ``require_netns=True`` and get a RuntimeError instead of a silent
  downgrade.

This is a robustness sandbox for model-written code, not a security boundary against a
deliberately adversarial human: a program could setsid() out of the kill group or find a
kernel bug. For eval/reward traffic from our own models the limits above are the failure
modes that actually occur (infinite loops, runaway allocation, output bombs, stray
network calls), and each one is hard-capped.
"""

from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MEMORY_MB = 512
DEFAULT_MAX_OUTPUT_BYTES = 65_536
# RLIMIT_FSIZE for anything the child writes (including its stdout/stderr files).
_MAX_FILE_BYTES = 8 << 20

_NETNS_SUPPORTED: bool | None = None  # probed once per process


@dataclass(frozen=True)
class ExecResult:
    """Outcome of one sandboxed run. ``exit_code`` is negative when a signal killed the
    program (-9 after our timeout SIGKILL, -25/SIGXFSZ on an output bomb, ...)."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float
    network_isolation: str  # "netns" | "env-only"

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def netns_available() -> bool:
    """True when `unshare -rn` (unprivileged user+net namespace) works on this kernel.
    Probed once and cached; the probe itself runs `true` inside the namespace."""
    global _NETNS_SUPPORTED
    if _NETNS_SUPPORTED is None:
        unshare = shutil.which("unshare")
        if unshare is None:
            _NETNS_SUPPORTED = False
        else:
            probe = subprocess.run(
                [unshare, "-r", "-n", "true"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
            )
            _NETNS_SUPPORTED = probe.returncode == 0
    return _NETNS_SUPPORTED


def _make_limit_setter(memory_mb: int, timeout_s: float):
    """preexec_fn run in the child between fork and exec. Rlimits survive exec and are
    inherited by grandchildren, so they hold for the whole process tree."""

    def set_limits() -> None:
        mem = memory_mb << 20
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        cpu = int(timeout_s) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (_MAX_FILE_BYTES, _MAX_FILE_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return set_limits


def _read_capped(path: Path, cap: int) -> str:
    data = path.read_bytes()
    if len(data) > cap:
        return data[:cap].decode("utf-8", errors="replace") + "\n...[truncated]"
    return data.decode("utf-8", errors="replace")


def run_python(
    code: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    memory_mb: int = DEFAULT_MEMORY_MB,
    python: str = sys.executable,
    require_netns: bool = False,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> ExecResult:
    """Run ``code`` as a Python program under the limits described in the module
    docstring. Never raises for anything the *program* does; raises RuntimeError only
    when ``require_netns=True`` and the kernel cannot provide a network namespace."""
    if require_netns and not netns_available():
        raise RuntimeError(
            "network-namespace isolation (unshare -rn) is unavailable on this kernel; "
            "refusing to run with require_netns=True"
        )
    isolation = "netns" if netns_available() else "env-only"

    with tempfile.TemporaryDirectory(prefix="microlab-exec-") as tmp:
        tmpdir = Path(tmp)
        prog = tmpdir / "prog.py"
        prog.write_text(code, encoding="utf-8")
        out_path = tmpdir / "stdout"
        err_path = tmpdir / "stderr"

        argv = [python, "-I", "-B", str(prog)]
        if isolation == "netns":
            argv = [shutil.which("unshare"), "-r", "-n", "--", *argv]
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmpdir),
            "TMPDIR": str(tmpdir),
            "LC_ALL": "C.UTF-8",
        }

        start = time.monotonic()
        timed_out = False
        with out_path.open("wb") as out_f, err_path.open("wb") as err_f:
            proc = subprocess.Popen(
                argv,
                cwd=tmpdir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out_f,
                stderr=err_f,
                start_new_session=True,
                preexec_fn=_make_limit_setter(memory_mb, timeout_s),
            )
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # exited between the timeout and the kill
                proc.wait()
        duration = time.monotonic() - start

        return ExecResult(
            exit_code=proc.returncode,
            stdout=_read_capped(out_path, max_output_bytes),
            stderr=_read_capped(err_path, max_output_bytes),
            timed_out=timed_out,
            duration_s=duration,
            network_isolation=isolation,
        )
