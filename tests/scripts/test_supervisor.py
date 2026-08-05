"""Supervisor logic: progress detection, cost accounting, and the resume contract.

The supervisor exists so a preemption costs six minutes instead of the run. Every test
here targets a way that promise could fail QUIETLY — a stalled box that still bills, a
resume that rewinds, a cost counter that forgets what earlier instances spent.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sup = _load("vast_supervisor")


class FakeS3:
    def __init__(self, keys):
        self._keys = keys

    def list_objects_v2(self, **kw):
        pfx = kw.get("Prefix", "")
        items = [{"Key": k, "Size": v} for k, v in self._keys.items() if k.startswith(pfx)]
        return {"Contents": items, "IsTruncated": False}


def test_progress_is_the_highest_step_not_the_newest_upload():
    """Re-uploading an older checkpoint must not rewind the run.

    B2 is object storage: a retried upload of ckpt_500 after ckpt_2000 exists would have a
    later mtime. Taking the newest object would silently resume 1,500 steps back and burn
    that compute again.
    """
    s3 = FakeS3({"coder-1b/ckpt_500.pt": 10, "coder-1b/ckpt_2000.pt": 10,
                 "coder-1b/ckpt_1000.pt": 10})
    assert sup.remote_step(s3, "b", "coder-1b") == 2000


def test_no_checkpoints_means_step_zero():
    assert sup.remote_step(FakeS3({}), "b", "coder-1b") == 0


def test_unrelated_objects_are_ignored():
    """The bucket also holds shakedown logs; they must not parse as progress."""
    s3 = FakeS3({"coder-1b/ckpt_100.pt": 1, "shakedown/shakedown-1.log": 1,
                 "coder-1b/tokenizer.json": 1})
    assert sup.remote_step(s3, "b", "coder-1b") == 100


def test_prefix_isolates_runs():
    """Two runs in one bucket must not read each other's progress."""
    s3 = FakeS3({"coder-1b/ckpt_100.pt": 1, "other-run/ckpt_9000.pt": 1})
    assert sup.remote_step(s3, "b", "coder-1b") == 100
    assert sup.remote_step(s3, "b", "other-run") == 9000


def test_spend_accumulates_across_episodes(tmp_path, monkeypatch):
    """Cost is cumulative over ALL instances. Resetting per instance would let an
    arbitrary number of preemptions each stay under the cap while the total ran away."""
    monkeypatch.setattr(sup, "STATE", tmp_path / "state.json")
    sup.save_state({"spent": 40.0, "episodes": [{"instance": 1}], "last_step": 3000})
    st = sup.load_state()
    st["spent"] += 35.0
    sup.save_state(st)
    assert sup.load_state()["spent"] == pytest.approx(75.0)
    assert sup.load_state()["last_step"] == 3000


def test_state_survives_a_supervisor_restart(tmp_path, monkeypatch):
    """The supervisor itself can be killed; the spend must not reset to zero."""
    monkeypatch.setattr(sup, "STATE", tmp_path / "state.json")
    sup.save_state({"spent": 123.45, "episodes": [], "last_step": 12000})
    assert json.loads((tmp_path / "state.json").read_text())["spent"] == 123.45
    assert sup.load_state()["spent"] == pytest.approx(123.45)


def test_missing_state_starts_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(sup, "STATE", tmp_path / "none.json")
    st = sup.load_state()
    assert st == {"spent": 0.0, "episodes": [], "last_step": 0}


def test_onstart_runs_the_training_entrypoint():
    class A:
        repo = "https://example.com/r.git"
    s = sup.onstart(A())
    assert "cloud_train.sh" in s and "git clone" in s


def test_checkpoint_syncer_never_prunes_the_newest():
    """The syncer must not delete the file the next instance would resume from."""
    src = (SCRIPTS / "b2_ckpt_sync.py").read_text()
    assert "if p == local[-1]:" in src, "newest-checkpoint guard is missing"
    assert "confirmed" in src, "pruning must be gated on remote confirmation"


def test_crash_detection_ignores_a_stale_log(monkeypatch):
    """A previous episode's traceback must not condemn a healthy new instance.

    The log key is reused across episodes and a fresh box takes minutes to ship its first
    log. Reading it unguarded destroyed a healthy instance on the previous run's evidence —
    a false positive that costs more than the gap it closes.
    """
    import datetime as dt

    class S3:
        def __init__(self, when):
            self.when = when

        def get_object(self, Bucket, Key):
            class B:
                @staticmethod
                def read():
                    return b"MICROLAB_TRAIN_FAILED rc=1"
            return {"Body": B(), "LastModified": self.when}

    old = dt.datetime(2020, 1, 1, tzinfo=dt.UTC)
    new = dt.datetime(2030, 1, 1, tzinfo=dt.UTC)
    episode_start = dt.datetime(2025, 1, 1, tzinfo=dt.UTC).timestamp()

    assert sup.training_crashed(S3(old), "b", "p", since=episode_start) is None, \
        "a log older than the episode must be ignored"
    assert sup.training_crashed(S3(new), "b", "p", since=episode_start) is not None, \
        "a log written during this episode must still be read"


def test_crash_detection_without_a_since_still_reads():
    """Backwards compatible: no episode start means no freshness filter."""
    import datetime as dt

    class S3:
        def get_object(self, Bucket, Key):
            class B:
                @staticmethod
                def read():
                    return b"MICROLAB_TRAIN_FAILED rc=2"
            return {"Body": B(), "LastModified": dt.datetime.now(dt.UTC)}

    assert sup.training_crashed(S3(), "b", "p") is not None


def _s3_with(text: str, when=None):
    import datetime as dt

    class S3:
        def get_object(self, Bucket, Key):
            class B:
                @staticmethod
                def read():
                    return text.encode()
            return {"Body": B(), "LastModified": when or dt.datetime.now(dt.UTC)}
    return S3()


def test_the_cap_counts_the_episode_that_is_running_right_now():
    """The cap was checked against BANKED spend, which only moves when an episode ends.

    A healthy run therefore compared a number that never changed. With --on-demand there
    is exactly one episode, so a 13-day, $500+ run could not trip a $250 cap at any point
    before finishing on its own. The projected figure was already printed on every poll;
    only the comparison used the stale one.
    """
    import time as _t
    st = {"spent": 100.0}
    one_hour_ago = _t.time() - 3600
    assert sup.spent_now(st, 2.0, one_hour_ago, 123) == pytest.approx(102.0, abs=0.05)


def test_spend_is_just_the_banked_total_when_nothing_is_rented():
    import time as _t
    st = {"spent": 100.0}
    assert sup.spent_now(st, None, None, None) == 100.0
    assert sup.spent_now(st, 2.0, _t.time(), None) == 100.0, "no instance, no accrual"
    assert sup.spent_now(st, 2.0, None, 123) == 100.0, "no episode start, no accrual"


def test_a_preempted_instance_is_destroyed_not_abandoned():
    """"Not running" is not "gone". The contract still bills for allocated disk, and Vast
    RESUMES an interruptible instance once the price falls back under the standing bid —
    the zombie then writes checkpoints and logs to the same prefix as the live box."""
    src = (SCRIPTS / "vast_supervisor.py").read_text()
    i = src.index("vanished — preempted")
    assert "vast.destroy(inst, key)" in src[i:i + 900], \
        "the preemption path must destroy the instance before re-provisioning"


def test_the_crash_path_does_not_bank_the_same_episode_twice():
    """It banked, then broke with t_ep still set, so the finally banked it again."""
    src = (SCRIPTS / "vast_supervisor.py").read_text()
    i = src.index("TRAINING CRASHED")
    window = src[max(0, i - 500):i]
    assert "t_ep = None" in window, "crash path must clear t_ep so the finally cannot re-bank"


def test_an_exited_container_is_not_alive():
    """Presence in the instance list is not liveness.

    Preemption leaves the instance LISTED with actual_status "exited". Counting that as
    alive kept the supervisor reporting a healthy run — and paying for it — for 11 minutes
    after the container had stopped, instead of re-provisioning at once.
    """
    assert "exited" not in sup.RUNNING_STATES
    assert "stopped" not in sup.RUNNING_STATES
    assert {"running", "loading"} <= sup.RUNNING_STATES


def test_on_demand_omits_the_price_field():
    """Sending a price is what makes a Vast contract interruptible; --on-demand works by
    omitting it. A regression here would silently sell the run back to preemption."""
    src = (SCRIPTS / "vast_supervisor.py").read_text()
    assert 'if not a.on_demand:' in src
    i = src.index("if not a.on_demand:")
    assert 'body["price"]' in src[i:i + 1100], "price must be set only in the bid branch"
    assert '"price": round(' not in src, "price must not be unconditional in the body"


def test_on_demand_caps_the_right_price_field():
    """--max-price means dph_total on-demand and min_bid interruptible. Capping min_bid
    while renting on-demand would let the actual rate exceed the cap."""
    src = (SCRIPTS / "vast_supervisor.py").read_text()
    assert 'o.get("dph_total")) if a.on_demand else' in src


def test_logged_step_reads_progress_from_the_shipped_log():
    """The watchdog needs a signal that moves on ITS timescale.

    This is the bug that burned real money: progress was measured only by checkpoints in
    B2, but ckpt_interval=250 at ~30 s/step puts the first one over two hours out. The
    30-minute setup grace therefore expired while the box was training at 100% GPU, and
    the supervisor destroyed it — then re-provisioned into the identical doomed wait.
    """
    log = ("=== train ===\nstep 25/40000 loss 10.7391 lr 1.07e-05\n"
           "step 50/40000 loss 9.9 lr 2e-05\n")
    assert sup.logged_step(_s3_with(log), "b", "coder-1b") == 50


def test_logged_step_is_zero_before_any_step_line():
    """Corpus streaming is not progress; the setup clock must still apply."""
    log = "=== train ===\n  [shard] train-00098.bin 400 MB in 27s\n"
    assert sup.logged_step(_s3_with(log), "b", "coder-1b") == 0


def test_logged_step_ignores_a_stale_episodes_log():
    """A previous box's step count must not read as this box being alive — otherwise a
    wedged instance inherits the dead one's progress and never trips the watchdog."""
    import datetime as dt

    old = dt.datetime(2020, 1, 1, tzinfo=dt.UTC)
    episode_start = dt.datetime(2025, 1, 1, tzinfo=dt.UTC).timestamp()
    log = "step 900/40000 loss 4.4 lr 1e-04\n"
    assert sup.logged_step(_s3_with(log, old), "b", "p", since=episode_start) == 0
    assert sup.logged_step(_s3_with(log), "b", "p", since=episode_start) == 900


def test_logged_step_survives_a_missing_log():
    class S3:
        def get_object(self, Bucket, Key):
            raise KeyError("NoSuchKey")

    assert sup.logged_step(S3(), "b", "p") == 0


def test_either_signal_moving_counts_as_progress():
    assert sup.made_progress((0, 0), (0, 25)), "a log step alone must reset the clock"
    assert sup.made_progress((0, 25), (250, 25)), "a checkpoint alone must reset it too"
    assert not sup.made_progress((250, 300), (250, 300)), "no movement is not progress"


def test_a_stalled_log_is_not_masked_by_a_high_checkpoint_step():
    """Compared component-wise on purpose. As raw tuples, (250, 300) > (250, 0) would let
    a run that has stopped logging keep resetting its own stall clock forever."""
    assert not sup.made_progress((250, 300), (250, 200)), \
        "a log that went BACKWARDS is not progress"


def test_the_trainer_emits_its_first_step_regardless_of_log_interval():
    """Both ends again: the supervisor cannot read a step line the trainer never prints.
    At log_interval=25 the first line is ~13 minutes out, which is longer than the stall
    timeout it is supposed to satisfy."""
    src = (Path(__file__).resolve().parents[2] / "src" / "microlab" / "train"
           / "trainer.py").read_text()
    assert "if due or not logged_any:" in src, "first-step liveness print is missing"


def test_benign_log_noise_is_not_a_crash():
    """The detector must not infer failure from prose.

    A heuristic scan for "Error" matched this exact line — optional NVML telemetry being
    absent, which the trainer handles and reports — and destroyed a healthy instance.
    """
    noise = ("GPU NVML telemetry unavailable (memory-only): ModuleNotFoundError: "
             "nvidia-ml-py does not seem to be installed or it can't be imported.\n"
             "Traceback (most recent call last):\n  handled during import probe\n"
             "step 5/40000 loss 8.1")
    assert sup.training_crashed(_s3_with(noise), "b", "p") is None


def test_explicit_sentinel_is_a_crash():
    log = "step 3/40000 loss 9.9\nMICROLAB_TRAIN_FAILED rc=1\n"
    got = sup.training_crashed(_s3_with(log), "b", "p")
    assert got and "MICROLAB_TRAIN_FAILED" in got and "rc=1" in got


def test_the_sentinel_is_actually_emitted_by_the_instance_script():
    """Both ends of the channel must agree, or the detector never fires at all."""
    src = (SCRIPTS / "cloud_train.sh").read_text()
    assert "MICROLAB_TRAIN_FAILED rc=$RC" in src
    assert 'if [ "$RC" -ne 0 ]' in src


def test_help_actually_renders():
    """argparse %-expands help strings, so a bare '%' in prose raises at --help time.

    An unescaped "0.7% to remove preemption" made --help crash with
    "unsupported format character 't'" — invisible to every run, because no run calls
    --help, and therefore exactly the kind of breakage that surfaces when someone is
    trying to find a flag under time pressure.
    """
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(SCRIPTS / "vast_supervisor.py"), "--help"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-800:]
    assert "--skip-corpus-check" in r.stdout


def test_renting_is_gated_on_the_corpus_assertions():
    """Deselected from the commit guardrail, mandatory before spending money."""
    src = (SCRIPTS / "vast_supervisor.py").read_text()
    assert "skip_corpus_check" in src
    i = src.index("skip_corpus_check")
    assert "test_mix_artifact.py" in src[i:i + 900], \
        "the gate must actually run the corpus assertions"
    assert src.index("skip_corpus_check") < src.index("inst = bid = None"), \
        "the gate must run BEFORE the provisioning loop"


def test_the_shard_prefix_reaches_the_instance():
    """One flag must move the whole run to a new corpus build. mix-v1 was hardcoded in
    three places in cloud_train.sh; a run pointed at mix-v2 would have silently streamed
    the OLD corpus while the config claimed the new one."""
    src = (SCRIPTS / "vast_supervisor.py").read_text()
    assert "SHARD_PREFIX=a.shard_prefix" in src
    sh = (SCRIPTS / "cloud_train.sh").read_text()
    assert "SHARD_PREFIX=${SHARD_PREFIX:-mix-v1}" in sh
    assert "mix-v1/{f}" not in sh, "manifest fetch must use $SHARD_PREFIX, not a literal"
    assert 'export MICROLAB_SHARD_PREFIX="$SHARD_PREFIX"' in sh


def test_the_config_rewrite_is_verified_not_assumed():
    """sed exits 0 on a no-match. An unmatched literal meant the trainer wrote to a path
    the syncer was not watching — a full rental producing nothing durable, invisible to
    the watchdog because the log keeps shipping. The rewrite must be proven by importing
    the config exactly as the trainer will."""
    sh = (SCRIPTS / "cloud_train.sh").read_text()
    assert "config rewrite did not take" in sh
    assert "MICROLAB_TRAIN_FAILED rc=97" in sh, \
        "a failed rewrite must emit the sentinel so the supervisor stops the run"
    i = sh.index("config rewrite did not take")
    assert "importlib.util" in sh[i - 1200:i], \
        "verification must import the config, not grep it"


def test_remote_prune_spares_milestones_and_the_rolling_window():
    """The bucket accumulated every 50-step checkpoint of a 2,200-step run — 405 GB of
    files nothing will resume from, growing to 7.4 TB over the full run. The prune must
    remove exactly the stale rolling checkpoints: never a milestone (the emergence
    trajectory), never the newest N (the recovery window)."""
    src = (SCRIPTS / "b2_ckpt_sync.py").read_text()
    assert "% a.milestone_interval == 0" in src, "milestone exemption missing"
    assert "rolled[:-a.remote_keep]" in src, "newest-N window missing"
    assert 'default=0' in src.split("--remote-keep")[1][:400], \
        "remote pruning must be opt-in — a default-on deleter is how trajectories vanish"


def test_remote_prune_is_wired_into_the_instance_with_matching_milestones():
    """Both ends must agree: the syncer's exemption interval has to match the trainer's
    ckpt_milestone_interval (2000 in coder-1b) or milestones get deleted as rolling."""
    sh = (SCRIPTS / "cloud_train.sh").read_text()
    assert "--remote-keep 3" in sh
    assert "--milestone-interval 2000" in sh
    cfg = (SCRIPTS.parent / "configs" / "coder-1b.py").read_text()
    assert "ckpt_milestone_interval=2000" in cfg


def test_a_resumed_episode_still_gets_its_setup_grace():
    """`started` judged by st["last_step"] > 0 denies every RESUMED episode its setup
    grace — last_step is always positive on a resume — so the 25-min stall clock killed a
    healthy box 25 minutes into a 35-minute Thailand setup, mid checkpoint-download. The
    boundary must be per-episode: this box's log, or the durable step ADVANCING past
    where the episode began."""
    src = (SCRIPTS / "vast_supervisor.py").read_text()
    assert "started = marker[1] > 0" in src, \
        "only THIS box's freshness-gated log may witness that training started — a " \
        "dying box's late checkpoint upload faked the checkpoint-advance arm and got " \
        "a healthy setup killed by the stall clock"
    assert 'started = marker[1] > 0 or st["last_step"] > 0\n' not in src, \
        "the global-progress test is the bug, not the fix"
    i = src.index("inst, bid = provision(a, key, creds)")
    assert "ep_start_step = st[\"last_step\"]" in src[i:i + 1600], \
        "each provision must record where its episode began"


def test_bids_are_sticky_but_never_exceed_the_cap():
    """floor+2% was sniped five times in a day, several mid-setup — dead spend each time.
    The bid is now floor+25%, and the cap still binds it absolutely."""
    src = (SCRIPTS / "vast_supervisor.py").read_text()
    assert "min(bid * 1.25, a.max_price * a.gpus)" in src
    assert "round(bid * 1.02, 4)" not in src, "the snipeable bid must be gone"


def test_compile_caches_round_trip_through_b2():
    """Every re-provision re-paid 15-20 min of max-autotune at full 4-GPU price. The
    inductor/triton caches are arch-keyed (all our hosts are sm90), so shipping them
    through B2 turns compile into ~2-3 min of validation. Both halves must exist:
    restore BEFORE training, ship AFTER autotune completes."""
    sh = (SCRIPTS / "cloud_train.sh").read_text()
    assert "TORCHINDUCTOR_CACHE_DIR" in sh and "TRITON_CACHE_DIR" in sh
    assert "caches/compile-cache.tar" in sh
    restore = sh.index("compile cache restored")
    train = sh.index('say "train: $NGPU GPU(s)"')
    ship = sh.index("compile cache shipped")
    assert restore < train, "restore must happen before training starts"
    assert "sleep 1500" in sh, "ship must wait for autotune to finish first"
    assert ship < train, "the shipping subshell must be launched before the blocking train"


def test_episode_logs_are_archived_not_destroyed():
    """A box that dies during setup leaves its shipped log as the ONLY post-mortem.
    The provision-time purge deleted it, so a $2 sixty-minute grace window ended in a
    mystery. Old logs move to logs/archive/epNNN-*, they do not vanish."""
    src = (SCRIPTS / "vast_supervisor.py").read_text()
    i = src.index("nothing stale to misread")
    block = src[i:i + 900]
    assert "/logs/archive/" in block, "old episode logs must be archived"
    assert "s3.copy(" in block, "archive must copy before deleting"


def test_setup_phase_reads_markers_and_progress():
    """During setup the step count is necessarily zero, so the poll line must say WHERE
    setup is. A box downloading torch at 11 MB/s and a bricked one used to produce the
    same line, on four billing GPUs."""
    log = ("=== environment ===\nblah\n=== dependencies ===\npip stuff\n"
           "=== resume: newest checkpoint from B2, if any ===\n"
           "  [resume] 40% 3.7/9.2 GB @ 12 MB/s\n")
    got = sup.setup_phase(_s3_with(log), "b", "p")
    assert "resume" in got and "12 MB/s" in got


def test_setup_phase_survives_a_missing_log():
    class S3:
        def get_object(self, Bucket, Key):
            raise KeyError("NoSuchKey")
    assert sup.setup_phase(S3(), "b", "p") == "no log yet"


def test_early_shipper_starts_before_the_slow_deps_phase():
    """The first log ship used to happen AFTER pip; on a slow pipe that was 30-45 min of
    'log 0' with four GPUs billing. The early shipper must be armed before deps and
    retire when the real syncer starts."""
    sh = (SCRIPTS / "cloud_train.sh").read_text()
    early = sh.index('say "early log shipping"')
    deps = sh.index('say "dependencies"')
    assert early < deps, "early shipper must precede the deps phase"
    assert ".syncer-started" in sh, "the real syncer must retire the early shipper"
    assert sh.index("touch \"$WORK/.syncer-started\"") > sh.index("b2_ckpt_sync.py --run")


def test_syncer_ships_logs_before_touching_checkpoints():
    """A checkpoint-path error must not starve the liveness signal. The trainer's own
    pruner deleted a file between the syncer's glob and stat; the pass aborted before
    the log-shipping section, and the shipped log froze for 30+ minutes while the box
    trained on — indistinguishable from a shipping outage."""
    src = (SCRIPTS / "b2_ckpt_sync.py").read_text()
    assert src.index("LOGS. Without these") < src.index("remote = b2.remote_sizes"), \
        "log shipping must run before checkpoint handling in each pass"
    i = src.index("for p in local:")
    assert "except FileNotFoundError:" in src[i:i + 700], \
        "a file the trainer pruned mid-pass must be skipped, not abort the pass"


def test_the_default_image_is_the_baked_stack():
    """The pip phase cost 8-10 min on good pipes and 25-100+ min on Asia hosts (three
    boxes died in setup on it). The baked image replaces it with a per-host-cached
    layer pull; the entrypoint's version asserts still gate a stale image loudly."""
    src = (SCRIPTS / "vast_supervisor.py").read_text()
    assert 'default="ghcr.io/rje/microlab-train:cu126-1"' in src
