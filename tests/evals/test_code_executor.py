"""Hard-limit tests for the sandboxed executor: these are the guarantees the whole code
eval (and later the GRPO reward) stands on — timeouts actually kill, the memory cap
actually caps, output is captured and bounded, the environment leaks nothing."""

from __future__ import annotations

import time

import pytest

from microlab.evals.code.executor import ExecResult, netns_available, run_python


def test_result_capture_correct():
    res = run_python(
        'import sys\nprint("out-line")\nprint("err-line", file=sys.stderr)\n'
    )
    assert res.passed
    assert res.exit_code == 0
    assert res.stdout == "out-line\n"
    assert res.stderr == "err-line\n"
    assert not res.timed_out
    assert res.network_isolation in ("netns", "env-only")


def test_nonzero_exit_code_captured():
    res = run_python("import sys\nsys.exit(3)\n")
    assert res.exit_code == 3
    assert not res.passed


def test_exception_surfaces_in_stderr():
    res = run_python('raise ValueError("boom-marker")\n')
    assert not res.passed
    assert res.exit_code == 1
    assert "boom-marker" in res.stderr
    assert "ValueError" in res.stderr


def test_timeout_kills_infinite_loop():
    start = time.monotonic()
    res = run_python("while True:\n    pass\n", timeout_s=1.5)
    elapsed = time.monotonic() - start
    assert res.timed_out
    assert not res.passed
    assert res.exit_code != 0
    assert elapsed < 10, "SIGKILL after the timeout must not hang"


def test_timeout_kills_forked_children_too():
    # The program forks; both processes spin. start_new_session + killpg must take the
    # whole group down, not just the parent.
    code = (
        "import os\n"
        "os.fork()\n"
        "while True:\n    pass\n"
    )
    start = time.monotonic()
    res = run_python(code, timeout_s=1.5)
    assert res.timed_out
    assert time.monotonic() - start < 10


def test_memory_cap_kills_allocator():
    res = run_python("x = bytearray(1 << 30)\nprint(len(x))\n", memory_mb=256)
    assert not res.passed
    assert res.exit_code != 0
    assert "MemoryError" in res.stderr
    assert res.stdout == ""


def test_memory_cap_allows_normal_work():
    res = run_python("x = list(range(100_000))\nprint(sum(x))\n", memory_mb=256)
    assert res.passed
    assert res.stdout.strip() == str(sum(range(100_000)))


def test_output_is_truncated_not_buffered():
    res = run_python('print("A" * 1000)\n' * 200, max_output_bytes=4096)
    assert res.passed
    assert res.stdout.endswith("...[truncated]")
    assert len(res.stdout) < 5000


def test_runs_in_isolated_tmpdir_which_is_cleaned():
    res = run_python(
        "import os\nopen('scratch.txt', 'w').write('x')\nprint(os.getcwd())\n"
    )
    assert res.passed
    cwd = res.stdout.strip()
    assert "microlab-exec-" in cwd
    import os

    assert not os.path.exists(cwd), "sandbox tmpdir must be deleted after the run"


def test_environment_is_whitelist_not_inherited(monkeypatch):
    monkeypatch.setenv("MICROLAB_CANARY_SECRET", "leak-me")
    res = run_python(
        "import os\nprint(sorted(os.environ))\nprint('HOME=' + os.environ['HOME'])\n"
    )
    assert res.passed
    assert "MICROLAB_CANARY_SECRET" not in res.stdout
    assert "PYTHONPATH" not in res.stdout
    assert "microlab-exec-" in res.stdout  # HOME points into the sandbox tmpdir


def test_require_netns_raises_or_isolates():
    if netns_available():
        res = run_python("print('ok')\n", require_netns=True)
        assert res.passed and res.network_isolation == "netns"
    else:
        with pytest.raises(RuntimeError, match="unavailable"):
            run_python("print('ok')\n", require_netns=True)


@pytest.mark.skipif(not netns_available(), reason="unprivileged netns not supported here")
def test_network_actually_blocked_under_netns():
    code = (
        "import socket\n"
        "s = socket.socket()\n"
        "s.settimeout(3)\n"
        "try:\n"
        "    s.connect(('1.1.1.1', 80))\n"
        "    print('CONNECTED')\n"
        "except OSError:\n"
        "    print('BLOCKED')\n"
    )
    res = run_python(code, require_netns=True)
    assert res.stdout.strip() == "BLOCKED"


def test_passed_property():
    ok = ExecResult(0, "", "", False, 0.1, "env-only")
    assert ok.passed
    assert not ExecResult(1, "", "", False, 0.1, "env-only").passed
    assert not ExecResult(0, "", "", True, 0.1, "env-only").passed
