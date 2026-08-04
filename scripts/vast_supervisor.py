#!/usr/bin/env python
"""Supervise a preemptible training run: provision, watch, re-provision, stop, destroy.

    python scripts/vast_supervisor.py --gpus 4 --max-price 0.60 --max-spend 250 \\
        --target-step 40000 --run-prefix coder-1b            # dry run
    python scripts/vast_supervisor.py ... --yes               # actually rents

Bid pricing is ~30% of on-demand, which is what makes a $600 run a $160 one, and the
trade is that the instance is taken away when someone outbids you. Over a multi-day run
that WILL happen. This process is what turns that from "the run silently stopped at hour
40" into "the run continued on a new box six minutes later".

WHAT IT IS NOT: it does not babysit training correctness. Progress is judged solely by
checkpoints arriving in B2 with increasing step numbers, because that is the only signal
that survives the instance vanishing.

MONEY. Every guard exists because the failure is silent and continuous:

  * --max-spend is a HARD cumulative cap across ALL instances this run has used. Reaching
    it destroys the instance and exits, whatever the training state.
  * --max-price is re-checked against the live offer before every provision, so a market
    move between attempts cannot quietly raise the burn rate.
  * a stall watchdog: if no new checkpoint appears within --stall-minutes, the instance is
    destroyed and replaced. A wedged box bills exactly like a working one.
  * the instance is destroyed in a `finally`, and the destroy is VERIFIED against the
    instance list — Vast's DELETE has already returned success while the box kept running.
  * on exit, any surviving instance is reported loudly with a link.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

_v = importlib.util.spec_from_file_location(
    "vast_run", Path(__file__).resolve().parent / "vast_run.py")
vast = importlib.util.module_from_spec(_v)
_v.loader.exec_module(vast)

_b = importlib.util.spec_from_file_location(
    "b2_sync", Path(__file__).resolve().parent / "b2_sync.py")
b2 = importlib.util.module_from_spec(_b)
_b.loader.exec_module(b2)

STATE = Path("runs/.supervisor-state.json")


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"spent": 0.0, "episodes": [], "last_step": 0}


def save_state(s: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=1))


def training_crashed(s3, bucket: str, prefix: str, since: float | None = None) -> str | None:
    """Detect a DEAD TRAINER on a LIVE box, from the log shipped to B2.

    The instance-liveness check cannot see this: a crashed trainer and a slow setup look
    identical from outside, so the run sat idle for 50 minutes after crashing 2 minutes in,
    and the setup-grace timer would eventually have re-provisioned into the same crash and
    looped. Returns the offending line, or None.
    """
    try:
        obj = s3.get_object(Bucket=bucket, Key=f"{prefix}/logs/train.log")
        # The log key is REUSED across episodes, so a previous instance's traceback sits
        # there until the new one ships its first log — several minutes into setup. Reading
        # it unguarded made the detector destroy a HEALTHY box on its own stale evidence.
        # Only a log written after this episode began describes this episode.
        if since is not None:
            mtime = obj["LastModified"].timestamp()
            if mtime < since:
                return None
        tail = obj["Body"].read().decode("utf-8", "replace")[-40_000:]
    except Exception:                               # noqa: BLE001
        return None                                 # no log yet is not a crash
    # ONE unambiguous sentinel, written by cloud_train.sh on a non-zero torchrun exit.
    # Deliberately not a heuristic: scanning for "Traceback" or "Error" matched a benign
    # "ModuleNotFoundError: nvidia-ml-py" telemetry notice and destroyed a healthy
    # instance. Inferring failure from prose is how a watchdog becomes the outage.
    if "MICROLAB_TRAIN_FAILED" in tail:
        for line in tail.splitlines():
            if "MICROLAB_TRAIN_FAILED" in line:
                return line.strip()[:200]
    return None


# Statuses that mean the container can still be making progress. Anything else — most
# importantly "exited", which is what preemption leaves behind — is a dead box.
RUNNING_STATES = frozenset({"running", "loading", "created", "starting"})


def logged_step(s3, bucket: str, prefix: str, since: float | None = None) -> int:
    """Highest step the shipped training log reports — the LIVENESS signal.

    Distinct from `remote_step`, and the two are not interchangeable. Checkpoints are
    durable but rare: at ckpt_interval=250 and ~30 s/step the first one is over two hours
    out, so a watchdog clocked on checkpoints alone declares any healthy run dead long
    before it can possibly show progress. That is not hypothetical — it destroyed a box
    that was training at 100% GPU, and re-provisioned into the same doomed wait.

    The log is shipped to B2 every 120 s, so this moves on the watchdog's timescale. It is
    NOT durable and must never be used to decide where to resume from: a step reported
    here has no checkpoint behind it.
    """
    try:
        obj = s3.get_object(Bucket=bucket, Key=f"{prefix}/logs/train.log")
        # Same freshness gate as the crash detector: the log key is reused across
        # episodes, and a previous episode's step count is not this box being alive.
        if since is not None and obj["LastModified"].timestamp() < since:
            return 0
        tail = obj["Body"].read().decode("utf-8", "replace")[-40_000:]
    except Exception:                               # noqa: BLE001
        return 0                                    # no log yet is not progress
    best = 0
    for line in tail.splitlines():
        if line.startswith("step "):
            try:
                best = max(best, int(line.split()[1].split("/")[0]))
            except (IndexError, ValueError):
                continue
    return best


def spent_now(st: dict, bid: float | None, t_ep: float | None, inst: int | None) -> float:
    """Banked spend PLUS what the running episode has cost so far.

    The cap used to be checked against banked spend alone, which is only incremented when
    an episode ENDS. A healthy run therefore tested a number that never moved: with
    --on-demand there is exactly one episode, so a 13-day, $500+ run could not trip a $250
    cap at any point. The projected figure was already being printed on every poll — only
    the comparison used the stale one.
    """
    if inst is None or t_ep is None or bid is None:
        return st["spent"]
    return st["spent"] + bid * (time.time() - t_ep) / 3600


def made_progress(prev: tuple[int, int], now: tuple[int, int]) -> bool:
    """Either signal moving forward counts. Compared component-wise, not as a tuple:
    tuple order would let a high checkpoint step mask a log that has stopped moving."""
    return now[0] > prev[0] or now[1] > prev[1]


def remote_step(s3, bucket: str, prefix: str) -> int:
    """Highest checkpoint step in B2 — the DURABLE signal, and the only one that outlives
    the box. Resume and target-reached decisions use this and nothing else."""
    best = 0
    for k in b2.remote_sizes(s3, bucket, f"{prefix}/ckpt_"):
        try:
            best = max(best, int(k.rsplit("ckpt_", 1)[1].split(".")[0]))
        except (IndexError, ValueError):
            continue
    return best


def onstart(a) -> str:
    return f"""
set -uo pipefail
cd /workspace 2>/dev/null || cd /root
git clone -q --depth 1 {a.repo} microlab || exit 1
bash microlab/scripts/cloud_train.sh 2>&1 | tee /workspace/train.log
"""


def provision(a, key, creds) -> tuple[int, float]:
    offers = vast.search_offers(a.gpu, a.max_price, a.min_reliability, a.min_disk,
                                a.gpus, key, by_bid=True,
                                verified=not a.allow_unverified)
    if a.host_id:
        offers = [o for o in offers if o.get("host_id") == a.host_id]
    if a.geo:
        offers = [o for o in offers if a.geo.lower() in str(o.get("geolocation", "")).lower()]
    # Which price the cap applies to, and what we are actually buying. On-demand is not
    # merely "safer" — on the host this was measured on it costs $1.49 against a $1.48
    # floor, a 0.7% premium that removes preemption entirely. Two runs died to preemption
    # before their first checkpoint, each losing 30+ steps of paid compute; that is far
    # more than 0.7%. Bidding is right for a long run that resumes cheaply, on-demand for
    # a short one that must finish in a single window.
    price_of = (lambda o: o.get("dph_total")) if a.on_demand else (lambda o: o.get("min_bid"))
    bids = []
    for o in offers:
        mb = price_of(o)
        if mb and mb / a.gpus <= a.max_price:
            bids.append((mb, o))
    if not bids:
        kind = "on-demand" if a.on_demand else "bid"
        raise SystemExit(
            f"no offer at <= ${a.max_price:.2f}/GPU-h {kind} for {a.gpus}x {a.gpu}. "
            f"Market moved; re-run later or raise --max-price.")
    bids.sort(key=lambda t: t[0])
    bid, pick = bids[0]
    env = dict(creds)
    env.update(REPO=a.repo, NGPU=str(a.gpus), CONFIG=a.config,
               RUN_PREFIX=a.run_prefix, BUCKET_IN=a.bucket_in, BUCKET_OUT=a.bucket_out,
               SHARD_PREFIX=a.shard_prefix)
    body = {"client_id": "me", "image": a.image, "disk": a.min_disk,
            "onstart": onstart(a), "env": env, "runtype": "ssh"}
    if not a.on_demand:
        # Sending a price is what makes the contract INTERRUPTIBLE. Omitting it rents
        # on-demand, which is the whole mechanism behind --on-demand.
        #
        # Bid floor+25%, capped at --max-price — NOT floor+2%. A hair-above-floor bid is
        # trivially sniped: five preemptions across two hosts in one day, several during
        # SETUP, each burning $0.40-0.75 of dead setup and a market slot. Paying up to
        # 25% over floor while running is cheaper than repeatedly paying full price for
        # boxes that die before their first step. Never exceeds the user-approved cap.
        body["price"] = round(min(bid * 1.25, a.max_price * a.gpus), 4)
    r = vast.call("PUT", f"/asks/{pick['id']}/", body, key=key)
    inst = r.get("new_contract")
    if not inst:
        raise SystemExit(f"create failed: {r}")
    print(f"  rented {inst} at ${bid:.2f}/h "
          f"({'on-demand' if a.on_demand else 'interruptible'}, {a.gpus}x "
          f"{pick.get('gpu_name')}, rel {pick.get('reliability2', 0):.3f}, "
          f"{pick.get('geolocation')})", flush=True)
    return inst, bid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--gpu", default="H100 SXM")
    ap.add_argument("--max-price", type=float, default=0.60, help="$/GPU-h bid ceiling")
    ap.add_argument("--max-spend", type=float, default=250.0, help="HARD cumulative cap")
    ap.add_argument("--target-step", type=int, required=True)
    ap.add_argument("--run-prefix", default="coder-1b")
    ap.add_argument("--config", default="configs/coder-1b.py")
    ap.add_argument("--repo", default="https://github.com/rje/microlab.git")
    ap.add_argument("--image", default="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    ap.add_argument("--min-reliability", type=float, default=0.97)
    ap.add_argument("--min-disk", type=int, default=120)
    ap.add_argument("--host-id", type=int, default=None,
                    help="restrict to this Vast host. Cheapest-first selection is right "
                         "for a multi-day run, where a slow corpus pull amortises away, "
                         "but wrong for a short test where the pull dominates wall-clock "
                         "— and price does not encode distance to the bucket.")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="include hosts Vast has not designated as datacenters. They are "
                         "the cheap tier and often fine, but reliability varies more — "
                         "pair with --min-reliability rather than trusting the label.")
    ap.add_argument("--on-demand", action="store_true",
                    help="rent NON-interruptible: --max-price then caps dph_total instead "
                         "of min_bid. Measured on the host used here, on-demand is $1.49 "
                         "against a $1.48 floor — 0.7%% to remove preemption, after two "
                         "runs were preempted before their first checkpoint and lost 30+ "
                         "steps each. Prefer it for short runs that must finish in one "
                         "window; bid for long runs that resume cheaply.")
    ap.add_argument("--skip-corpus-check", action="store_true",
                    help="rent even if tests/data/test_mix_artifact.py fails. Only "
                         "with a stated reason: those assertions exist because a val "
                         "set that was 100%% one repository passed every other check.")
    ap.add_argument("--geo", default=None,
                    help="substring the offer's geolocation must contain, e.g. 'US'")
    ap.add_argument("--bucket-in", default="microlab-corpus")
    ap.add_argument("--shard-prefix", default="mix-v1",
                    help="corpus prefix inside --bucket-in. The instance streams "
                         "shards from here AND the config data_dir is rewritten to "
                         "its local mirror, so one flag moves the whole run to a "
                         "new corpus build.")
    ap.add_argument("--bucket-out", default="microlab-checkpoints")
    ap.add_argument("--stall-minutes", type=int, default=90,
                    help="no new checkpoint in this long AFTER the first one -> replace")
    ap.add_argument("--setup-grace-minutes", type=int, default=120,
                    help="grace before the FIRST checkpoint. Setup legitimately produces "
                         "no checkpoints for a long time: a 39 GB corpus pull is ~10 min "
                         "from US-West and ~53 min from Asia, plus deps and compile. "
                         "Sharing one timer with the steady-state stall check made the "
                         "watchdog fire mid-download, destroy the box, and re-provision "
                         "onto an EMPTY disk that restarted the same download — an "
                         "infinite loop that only the spend cap stopped.")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    key = vast.api_key()
    creds = {}
    for name, pfx in (("b2_corpus", "B2_CORPUS"), ("b2_checkpoints", "B2_CKPT")):
        p = Path.home() / ".config" / "microlab" / f"{name}.env"
        if not p.exists():
            raise SystemExit(f"{p} not found")
        d = {}
        for line in p.read_text().splitlines():
            if "=" in line and line.strip().startswith("B2_"):
                k, v = line.split("=", 1)
                d[k.strip()] = v.strip()
        creds[f"{pfx}_KEY_ID"] = d["B2_KEY_ID"]
        creds[f"{pfx}_APPLICATION_KEY"] = d.get("B2_APPLICATION_KEY", d.get("B2_APP_KEY", ""))
        creds[f"{pfx}_ENDPOINT"] = d["B2_ENDPOINT"]

    s3 = b2.client(b2.load_credentials(b2.credential_path(a.bucket_out, None)))
    st = load_state()
    step = remote_step(s3, a.bucket_out, a.run_prefix)
    st["last_step"] = max(st["last_step"], step)
    print(f"target step {a.target_step:,}; B2 currently holds step {step:,}")
    print(f"prior spend on this run: ${st['spent']:.2f} of ${a.max_spend:.2f} cap")
    if step >= a.target_step:
        print("already complete.")
        return 0
    if not a.yes:
        print(f"\ndry run: would rent {a.gpus}x {a.gpu} at <= ${a.max_price:.2f}/GPU-h, "
              f"replacing on preemption or a {a.stall_minutes}-minute stall, "
              f"until step {a.target_step:,} or ${a.max_spend:.2f}.\nRe-run with --yes.")
        return 0

    # SPEND GATE. The corpus assertions are deselected from the commit guardrail because a
    # known-stale corpus must not block unrelated commits — but they absolutely must block
    # RENTING. A val set that was one geological-mesh repository survived every count-based
    # check and would have invalidated a $500 run whose only live metric reads from it.
    if not a.skip_corpus_check:
        import subprocess
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "corpus",
                            "tests/data/test_mix_artifact.py"],
                           capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        if r.returncode != 0:
            raise SystemExit(
                "corpus assertions FAILED — refusing to rent hardware to train on it:\n"
                f"{r.stdout[-2000:]}\n"
                "Rebuild the mix, or pass --skip-corpus-check if you know why this is fine.")
        print("corpus assertions pass", flush=True)

    inst = bid = None
    t_ep = None
    marker = (st["last_step"], 0)          # (durable checkpoint step, logged step)
    ep_start_step = st["last_step"]
    try:
        while st["last_step"] < a.target_step:
            if spent_now(st, bid, t_ep, inst) >= a.max_spend:
                print(f"\nHARD CAP: ${spent_now(st, bid, t_ep, inst):.2f} >= "
                      f"${a.max_spend:.2f}. Stopping at step {st['last_step']:,}. Re-run "
                      f"with a higher --max-spend to continue from this checkpoint.")
                break
            if inst is None:
                print(f"\nprovisioning (spent ${st['spent']:.2f}, "
                      f"step {st['last_step']:,}/{a.target_step:,})", flush=True)
                # Belt and braces with the freshness gate: move the previous episode's
                # log aside so there is nothing stale to misread even if clocks disagree.
                # ARCHIVE, not delete: a box that died during setup leaves the log as its
                # only post-mortem, and purging it here destroyed the evidence for a
                # $2 sixty-minute grace window that ended in a mystery.
                ep_n = len(st["episodes"])
                for k in list(b2.remote_sizes(s3, a.bucket_out, f"{a.run_prefix}/logs")):
                    if "/logs/archive/" in k:
                        continue                    # already archived
                    try:
                        dest = k.replace("/logs/", f"/logs/archive/ep{ep_n:03d}-", 1)
                        s3.copy({"Bucket": a.bucket_out, "Key": k}, a.bucket_out, dest)
                        s3.delete_object(Bucket=a.bucket_out, Key=k)
                    except Exception:               # noqa: BLE001
                        pass
                inst, bid = provision(a, key, creds)
                t_ep = time.time()
                last_progress = time.time()
                # Reset the LIVENESS half only. The new box starts its log at step 0 even
                # when resuming from a checkpoint, so carrying the old log high-water mark
                # over would make every subsequent poll look like no progress.
                marker = (st["last_step"], 0)
                # Where THIS episode began. The setup-vs-stall decision must compare
                # against this, not against zero: judging "training has started" by
                # st["last_step"] > 0 meant every RESUMED episode skipped its setup grace
                # entirely — last_step is always positive on a resume — and the stall
                # clock killed a healthy Thailand box 25 minutes into a setup that its
                # slow B2 pipe made 35 minutes long.
                ep_start_step = st["last_step"]
                st["episodes"].append({"instance": inst, "bid": bid,
                                       "from_step": st["last_step"]})
                save_state(st)

            time.sleep(a.poll)
            elapsed_h = (time.time() - t_ep) / 3600
            # Presence in the list is NOT liveness. A preempted box stays listed with
            # actual_status "exited", and treating that as alive kept the supervisor
            # reporting a healthy run — and paying for it — for 11 minutes after the
            # container had stopped, instead of re-provisioning immediately.
            alive = any(i.get("id") == inst and i.get("actual_status") in RUNNING_STATES
                        for i in vast.live_instances(key))
            now = remote_step(s3, a.bucket_out, a.run_prefix)
            live = logged_step(s3, a.bucket_out, a.run_prefix, since=t_ep)
            if now > st["last_step"]:
                st["last_step"] = now
            if made_progress(marker, (now, live)):
                marker = (max(marker[0], now), max(marker[1], live))
                last_progress = time.time()

            print(f"  [{elapsed_h*60:>5.0f}m] ckpt {st['last_step']:>7,}  "
                  f"log {live:>7,}  ${spent_now(st, bid, t_ep, inst):>7.2f}  "
                  f"{'alive' if alive else 'GONE'}", flush=True)

            # Re-check the cap HERE, against the running episode. The check at the top of
            # the loop only sees banked spend, so on a single long episode — exactly what
            # --on-demand produces — it would not fire until the run ended on its own.
            if spent_now(st, bid, t_ep, inst) >= a.max_spend:
                st["spent"] = spent_now(st, bid, t_ep, inst)
                save_state(st)
                print(f"\nHARD CAP mid-episode: ${st['spent']:.2f} >= ${a.max_spend:.2f} "
                      f"— destroying {inst}", flush=True)
                try:
                    vast.destroy(inst, key)
                except SystemExit as e:
                    print(f"  {e}")
                inst = t_ep = None
                break

            crash = training_crashed(s3, a.bucket_out, a.run_prefix, since=t_ep)
            if crash and st["last_step"] == 0:
                # A crash BEFORE any progress will repeat on a fresh box. Re-provisioning
                # into the same failure is the loop the spend cap exists to stop, so stop
                # here instead and surface the error.
                st["spent"] += bid * elapsed_h
                save_state(st)
                # Clear t_ep, NOT inst: the finally still has to destroy the box, but it
                # must not bank this episode a second time. (The stall and preempt paths
                # clear inst as well, so only this one double-counted.)
                t_ep = None
                print("\n  TRAINING CRASHED — not re-provisioning into the same failure:")
                print(f"    {crash}")
                break

            if not alive:
                # Preemption (or host failure). Bank the spend and re-provision; the next
                # box resumes from whatever reached B2.
                st["spent"] += bid * elapsed_h
                save_state(st)
                print(f"  instance {inst} vanished — preempted. Re-provisioning.",
                      flush=True)
                # DESTROY it. "Not running" is not "gone": the contract still exists and
                # still bills for its allocated disk, and Vast RESUMES an interruptible
                # instance by itself once the market price falls back under the standing
                # bid. A resumed zombie re-runs onstart and writes checkpoints and logs to
                # the SAME prefix as the live box — two trajectories interleaved under one
                # name, with remote_step taking the max across both.
                try:
                    vast.destroy(inst, key)
                except SystemExit as e:
                    print(f"  {e}")
                inst = t_ep = None
                continue

            # Two different clocks, because "no checkpoint yet" and "checkpoints stopped"
            # are different failures with very different legitimate durations.
            # "THIS EPISODE'S training has started" is the boundary, and it must be
            # per-episode: this box's log showing a step, or the durable checkpoint step
            # advancing past where the episode began. Two prior versions were each wrong
            # in one direction — gating on the checkpoint step alone kept fresh runs in
            # "setup" for the two hours before their first checkpoint, and gating on
            # st["last_step"] > 0 denied every RESUMED episode its setup grace, killing a
            # healthy box mid-download 25 minutes into a 35-minute setup.
            started = marker[1] > 0 or st["last_step"] > ep_start_step
            limit = a.stall_minutes if started else a.setup_grace_minutes
            phase = "stall" if started else "setup"
            if (time.time() - last_progress) / 60 > limit:
                # A wedged box bills identically to a working one; the only defence is a
                # clock on progress.
                st["spent"] += bid * elapsed_h
                save_state(st)
                print(f"  NO PROGRESS for {limit} min ({phase} phase) — destroying "
                      f"{inst}", flush=True)
                try:
                    vast.destroy(inst, key)
                except SystemExit as e:
                    print(f"  {e}")
                inst = None
                continue
        else:
            print(f"\nTARGET REACHED: step {st['last_step']:,}")
    finally:
        if inst:
            if t_ep:
                st["spent"] += bid * (time.time() - t_ep) / 3600
            save_state(st)
            print(f"\ndestroying {inst} …")
            try:
                vast.destroy(inst, key)
                print("destroyed")
            except SystemExit as e:
                print(f"DESTROY FAILED: {e}\n  -> kill it NOW: "
                      f"https://cloud.vast.ai/instances/")
        save_state(st)
        # Only STILL-RUNNING instances are a problem to shout about. Counting "exited"
        # ones here would cry wolf on every clean preemption and train the reader to
        # ignore the one message that means money is leaking.
        left = [i for i in vast.live_instances(key)
                if i.get("actual_status") in RUNNING_STATES]
        print(f"total spend ~${st['spent']:.2f}; step {st['last_step']:,}")
        if left:
            print(f"WARNING: {len(left)} instance(s) STILL RUNNING: "
                  f"{[i.get('id') for i in left]} — https://cloud.vast.ai/instances/")
        else:
            print("confirmed: no live instances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
