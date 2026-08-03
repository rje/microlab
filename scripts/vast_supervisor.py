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


def remote_step(s3, bucket: str, prefix: str) -> int:
    """Highest checkpoint step in B2 — the only progress signal that outlives the box."""
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
                                a.gpus, key, by_bid=True)
    bids = []
    for o in offers:
        mb = o.get("min_bid")
        if mb and mb / a.gpus <= a.max_price:
            bids.append((mb, o))
    if not bids:
        raise SystemExit(
            f"no offer at <= ${a.max_price:.2f}/GPU-h bid for {a.gpus}x {a.gpu}. "
            f"Market moved; re-run later or raise --max-price.")
    bids.sort(key=lambda t: t[0])
    bid, pick = bids[0]
    env = dict(creds)
    env.update(REPO=a.repo, NGPU=str(a.gpus), CONFIG=a.config,
               RUN_PREFIX=a.run_prefix, BUCKET_IN=a.bucket_in, BUCKET_OUT=a.bucket_out)
    r = vast.call("PUT", f"/asks/{pick['id']}/", {
        "client_id": "me", "image": a.image, "disk": a.min_disk,
        "onstart": onstart(a), "env": env, "runtype": "ssh",
        "price": round(bid * 1.02, 4),   # a hair above the floor, still under the cap
    }, key=key)
    inst = r.get("new_contract")
    if not inst:
        raise SystemExit(f"create failed: {r}")
    print(f"  rented {inst} at ${bid:.2f}/h ({a.gpus}x {pick.get('gpu_name')}, "
          f"rel {pick.get('reliability2', 0):.3f}, {pick.get('geolocation')})", flush=True)
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
    ap.add_argument("--bucket-in", default="microlab-corpus")
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

    inst = bid = None
    t_ep = None
    try:
        while st["last_step"] < a.target_step:
            if st["spent"] >= a.max_spend:
                print(f"\nHARD CAP: ${st['spent']:.2f} >= ${a.max_spend:.2f}. Stopping at "
                      f"step {st['last_step']:,}. Re-run with a higher --max-spend to "
                      f"continue from this checkpoint.")
                break
            if inst is None:
                print(f"\nprovisioning (spent ${st['spent']:.2f}, "
                      f"step {st['last_step']:,}/{a.target_step:,})", flush=True)
                inst, bid = provision(a, key, creds)
                t_ep = time.time()
                last_progress = time.time()
                st["episodes"].append({"instance": inst, "bid": bid,
                                       "from_step": st["last_step"]})
                save_state(st)

            time.sleep(a.poll)
            elapsed_h = (time.time() - t_ep) / 3600
            alive = any(i.get("id") == inst for i in vast.live_instances(key))
            now = remote_step(s3, a.bucket_out, a.run_prefix)
            if now > st["last_step"]:
                st["last_step"] = now
                last_progress = time.time()

            print(f"  [{elapsed_h*60:>5.0f}m] step {st['last_step']:>7,}  "
                  f"${st['spent'] + bid*elapsed_h:>7.2f}  "
                  f"{'alive' if alive else 'GONE'}", flush=True)

            if not alive:
                # Preemption (or host failure). Bank the spend and re-provision; the next
                # box resumes from whatever reached B2.
                st["spent"] += bid * elapsed_h
                save_state(st)
                print(f"  instance {inst} vanished — preempted. Re-provisioning.",
                      flush=True)
                inst = None
                continue

            # Two different clocks, because "no checkpoint yet" and "checkpoints stopped"
            # are different failures with very different legitimate durations.
            limit = a.stall_minutes if st["last_step"] > 0 else a.setup_grace_minutes
            phase = "stall" if st["last_step"] > 0 else "setup"
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
        left = vast.live_instances(key)
        print(f"total spend ~${st['spent']:.2f}; step {st['last_step']:,}")
        if left:
            print(f"WARNING: {len(left)} instance(s) STILL RUNNING: "
                  f"{[i.get('id') for i in left]} — https://cloud.vast.ai/instances/")
        else:
            print("confirmed: no live instances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
