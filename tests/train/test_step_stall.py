"""A hung training step must report its own stack and die.

A rented H100 froze at step 10 with 103 of 105 shards fetched, no error in the log, and
the provider still reporting 100% GPU utilisation. It billed at full rate until a human
noticed, and left nothing behind to diagnose from: the failure did not reproduce locally,
so there was no stack, no thread state, and no way to tell a wedged fetch from a wedged
kernel. These tests pin the two properties that turn that into evidence:

  * EVERY thread's stack is dumped, not just the main one — the suspect is a background
    shard-prefetch thread, which a main-thread traceback would not show at all;
  * the process EXITS non-zero, so cloud_train.sh emits MICROLAB_TRAIN_FAILED and the
    supervisor re-provisions from the last checkpoint. Warning and continuing would leave
    the box billing forever, which is the behaviour being fixed.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

from microlab.train.config import RunConfig


def _run(script: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(script)],
                          capture_output=True, text=True, timeout=timeout)


def test_a_hung_step_dumps_every_thread_and_exits_nonzero():
    p = _run("""
        import faulthandler, threading, time
        faulthandler.enable()

        def stuck_prefetch(ev):
            ev.wait()                       # a shard fetch that never returns

        ev = threading.Event()
        threading.Thread(target=stuck_prefetch, args=(ev,), daemon=True).start()
        time.sleep(0.2)
        faulthandler.dump_traceback_later(1.0, exit=True)
        time.sleep(30)                      # the step that never completes
    """)
    assert p.returncode != 0, "a stalled step must be fatal, not a warning"
    assert "Timeout" in p.stderr, p.stderr[:400]
    assert "stuck_prefetch" in p.stderr, (
        "background thread stack missing — a main-thread dump would not have found the "
        f"hang this exists for:\n{p.stderr[:600]}")


def test_a_step_that_finishes_in_time_is_not_killed():
    """The detector must not fire on a merely slow step, or it becomes the outage."""
    p = _run("""
        import faulthandler, time
        faulthandler.dump_traceback_later(10.0, exit=True)
        time.sleep(0.3)                     # a step, comfortably inside the budget
        faulthandler.cancel_dump_traceback_later()
        print("survived")
    """)
    assert p.returncode == 0, p.stderr[:400]
    assert "survived" in p.stdout


def test_rearming_measures_one_step_not_the_whole_run():
    """Armed per iteration. A single timer spanning many steps would fire mid-run on a
    perfectly healthy job once cumulative time passed the threshold."""
    p = _run("""
        import faulthandler, time
        for _ in range(6):
            faulthandler.dump_traceback_later(2.0, exit=True)
            time.sleep(0.25)
            faulthandler.cancel_dump_traceback_later()
        print("six steps, none tripped")
    """)
    assert p.returncode == 0, p.stderr[:400]
    assert "six steps" in p.stdout


def test_detector_is_off_by_default():
    """Opt-in: an unrelated local run must not acquire a fatal timer it never asked for."""
    assert RunConfig(block_size=8, batch_size=1).step_timeout_s == 0.0


def test_the_trainer_arms_around_the_step_only():
    """Eval and a multi-GB checkpoint write are legitimately slower than a step; arming
    across them would kill healthy runs at every checkpoint."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "src" / "microlab" / "train"
           / "trainer.py").read_text()
    arm = src.index("faulthandler.dump_traceback_later(stall_s, exit=True)")
    step = src.index("loss = self.train_step()", arm)
    cancel = src.index("faulthandler.cancel_dump_traceback_later()", arm)
    assert arm < step < cancel, "the timer must bracket train_step and nothing else"
