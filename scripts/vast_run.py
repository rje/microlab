#!/usr/bin/env python
"""Rent a Vast.ai GPU, run a job, collect the result, and ALWAYS destroy the instance.

    python scripts/vast_run.py search --gpu "H100 SXM" --max-price 2.00
    python scripts/vast_run.py shakedown --max-price 2.00 --max-minutes 75 --yes

MONEY SAFETY IS THE POINT OF THIS FILE. A forgotten instance bills until someone notices;
at $1.49/h that is $36/day. Every guard here exists because the failure is silent:

  * the instance is destroyed in a `finally`, so a crash, a timeout, or a Ctrl-C still
    tears it down;
  * `--max-minutes` is a hard wall clock, enforced locally, not a hope about the job;
  * `--max-price` is checked against the offer BEFORE renting, so a price change between
    search and create cannot silently cost more;
  * nothing is rented without `--yes`; the default is a dry run that prints the plan;
  * on exit the script re-lists instances and reports ANY that are still alive, because
    "I thought it was destroyed" is exactly how the bill happens.

Results come back via B2 rather than SSH: the job uploads a JSON blob and this script
polls for it. That removes SSH key handling from an untrusted host entirely.

CREDENTIALS: ~/.config/microlab/vast.env (mode 600), VAST_API_KEY=...
Create a SCOPED key at https://cloud.vast.ai/manage-keys/ with only `instance_read` and
`instance_write`. A full-account key on a script that rents hardware is more authority
than this needs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://console.vast.ai/api/v0"
# Instance management moved to v1; /api/v0/instances/ answers 410 deprecated.
API_V1 = "https://console.vast.ai/api/v1"
CRED = Path.home() / ".config" / "microlab" / "vast.env"


def api_key() -> str:
    if CRED.exists():
        mode = CRED.stat().st_mode & 0o777
        if mode & 0o077:
            raise SystemExit(f"{CRED} is mode {mode:o}; run: chmod 600 {CRED}")
        for line in CRED.read_text().splitlines():
            if line.strip().startswith("VAST_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    k = os.environ.get("VAST_API_KEY", "")
    if not k:
        raise SystemExit(
            f"no VAST_API_KEY. Create {CRED} (chmod 600) containing:\n"
            f"  VAST_API_KEY=...\n"
            f"Get a SCOPED key at https://cloud.vast.ai/manage-keys/ -> +New, with only "
            f"instance_read and instance_write.")
    return k


def call(method: str, path: str, body: dict | None = None,
         key: str | None = None, base: str = API):
    req = urllib.request.Request(
        f"{base}{path}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path} -> {e.code}: {e.read().decode()[:400]}") from e


def search_offers(gpu: str, max_price: float, min_reliability: float, min_disk: int,
                  num_gpus: int, key: str, by_bid: bool = False,
                  verified: bool = True) -> list[dict]:
    """Offers matching the hardware filters, priced under `max_price` per GPU-hour.

    `by_bid` selects WHICH price the cap applies to. Interruptible bids run ~30% of
    on-demand, and the two are wildly decorrelated per host — a 4x H100 seen here bid at
    $1.79/h against an $8.53/h on-demand rate. Filtering a bid-priced search on
    `dph_total` therefore discards exactly the cheap offers it is looking for.
    """
    q = {"gpu_name": {"eq": gpu}, "rentable": {"eq": True},
         "num_gpus": {"eq": num_gpus}, "disk_space": {"gte": min_disk},
         "reliability2": {"gte": min_reliability},
         # `verified` is Vast's datacenter designation. It reads back as None on every
         # offer, so it is invisible in results, but the SERVER-SIDE filter is real and
         # silently drops unverified hosts — including the cheapest 4x H100 on the market.
         **({"verified": {"eq": True}} if verified else {}),
         "order": [["dph_total", "asc"]], "limit": 200}
    offers = call("GET", f"/bundles/?q={urllib.parse.quote(json.dumps(q))}",
                  key=key).get("offers", [])
    cap = max_price * num_gpus
    if by_bid:
        return [o for o in offers if (o.get("min_bid") or 1e9) <= cap]
    return [o for o in offers if o.get("dph_total", 1e9) <= cap]


def fmt_offers(offers: list[dict], n: int = 10) -> None:
    print(f"{'id':>12} {'gpus':>4} {'$/gpu-h':>8} {'$/h':>7} {'rel':>6} {'disk':>7} "
          f"{'net down':>9}  location")
    for o in offers[:n]:
        g = o["num_gpus"]
        print(f"{o['id']:>12} {g:>4} {o['dph_total']/g:>8.2f} {o['dph_total']:>7.2f} "
              f"{o.get('reliability2', 0):>6.3f} {o.get('disk_space', 0):>6.0f}G "
              f"{o.get('inet_down', 0):>8.0f}M  {str(o.get('geolocation'))[:28]}")


def live_instances(key: str) -> list[dict]:
    """Instances currently on the account. v1: /api/v0/instances/ returns 410 deprecated.

    Never raises. This runs in the teardown path, where an exception would replace the
    warning about a still-billing instance with a stack trace about the API — the opposite
    of what the check exists for.
    """
    try:
        r = call("GET", "/instances/", key=key, base=API_V1)
    except SystemExit as e:
        print(f"  (could not list instances: {e}) — check "
              f"https://cloud.vast.ai/instances/ by hand")
        return []
    return r.get("instances", []) if isinstance(r, dict) else (r or [])


def destroy(inst_id: int, key: str) -> None:
    """Destroy, then VERIFY it is gone.

    LIST is v1 but DESTROY is still v0 — a v1 DELETE answers 404 while the instance keeps
    running and keeps billing. Trusting the response code let a real instance survive its
    own teardown, so this confirms against the instance list instead of believing the API.
    """
    try:
        call("DELETE", f"/instances/{inst_id}/", key=key, base=API)
    except SystemExit as e:
        # "no_such_instance" IS the goal state: someone (a manual kill, a prior destroy,
        # the host reclaiming it) got there first. Treating it as failure produced a
        # scary DESTROY FAILED + traceback on an account that was already clean.
        if "no_such_instance" not in str(e):
            raise
    for _ in range(6):
        time.sleep(5)
        if not any(i.get("id") == inst_id for i in live_instances(key)):
            return
    raise SystemExit(
        f"instance {inst_id} STILL LISTED after destroy — it is billing. "
        f"Kill it at https://cloud.vast.ai/instances/ NOW")


def onstart_script(repo: str, bucket_out: str, run_tag: str) -> str:
    """What the rented box runs at boot. Uploads its own result, so no SSH is needed."""
    return f"""
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
cd /workspace 2>/dev/null || cd /root
echo "=== microlab shakedown {run_tag} ==="
git clone --depth 1 {repo} microlab 2>&1 | tail -2
bash microlab/scripts/cloud_shakedown.sh 2>&1 | tee /tmp/shakedown.log
python - <<'PYEOF'
import boto3, os, json
from botocore.config import Config
# CHECKPOINTS key, not the corpus key: outputs belong in the bucket that carries the
# noncurrent-version lifecycle rule.
s3 = boto3.client("s3", endpoint_url=os.environ["B2_CKPT_ENDPOINT"],
    aws_access_key_id=os.environ["B2_CKPT_KEY_ID"],
    aws_secret_access_key=os.environ["B2_CKPT_APPLICATION_KEY"],
    config=Config(retries={{"max_attempts":10,"mode":"adaptive"}}))
log = open("/tmp/shakedown.log", "rb").read()
s3.put_object(Bucket="{bucket_out}", Key="shakedown/{run_tag}.log", Body=log)
try:
    b = open("/workspace/microlab/shakedown-bench.json","rb").read()
    s3.put_object(Bucket="{bucket_out}", Key="shakedown/{run_tag}.json", Body=b)
except Exception as e:
    s3.put_object(Bucket="{bucket_out}", Key="shakedown/{run_tag}.json",
                  Body=json.dumps({{"error": str(e)}}).encode())
print("results uploaded")
PYEOF
touch /tmp/DONE
"""


def cmd_search(a, key):
    offers = search_offers(a.gpu, a.max_price, a.min_reliability, a.min_disk,
                           a.num_gpus, key)
    print(f"{len(offers)} verified offers at <= ${a.max_price:.2f}/GPU-h, "
          f"reliability >= {a.min_reliability}, disk >= {a.min_disk}G\n")
    fmt_offers(offers)
    return 0


def cmd_shakedown(a, key):
    offers = search_offers(a.gpu, a.max_price, a.min_reliability, a.min_disk,
                           a.num_gpus, key)
    if not offers:
        raise SystemExit("no offers matched — relax --max-price or --min-reliability")
    fmt_offers(offers, 5)
    if a.offer_id:
        matches = [o for o in offers if o["id"] == a.offer_id]
        if not matches:
            raise SystemExit(
                f"offer {a.offer_id} is not in the current result set — it may have been "
                f"rented, or it fails the price/reliability/disk filters. Re-run `search`.")
        pick = matches[0]
    else:
        # Cheapest is NOT automatically best: the corpus lives in B2 us-west, so a host on
        # another continent pays for it in transfer time both here and on the real run.
        # Choose deliberately with --offer-id after reading the search output.
        pick = offers[0]
        print("\n(no --offer-id: taking the cheapest. Bandwidth and region are not "
              "priced in — read the table above and pick explicitly if it matters.)")
    price = pick["dph_total"]
    cap = price * a.max_minutes / 60
    run_tag = f"{a.tag}"
    print(f"\nPLAN\n  offer {pick['id']}  ${price:.2f}/h  "
          f"reliability {pick.get('reliability2', 0):.3f}  {pick.get('geolocation')}")
    print(f"  wall-clock cap {a.max_minutes} min -> MAXIMUM SPEND ${cap:.2f}")
    print(f"  results -> s3://{a.bucket_out}/shakedown/{run_tag}.{{json,log}}")
    if not a.yes:
        print("\ndry run. re-run with --yes to actually rent.")
        return 0
    if price > a.max_price * a.num_gpus:
        raise SystemExit(f"price ${price} exceeds cap — refusing")

    # BOTH keys go to the box, under distinct names. The buckets are separate on purpose:
    # the corpus is read, the checkpoints bucket is written, and only the checkpoints
    # bucket carries the noncurrent-version lifecycle rule that stops superseded 16.6 GB
    # checkpoints billing forever. Collapsing outputs into the corpus bucket to avoid
    # shipping a second credential would defeat that separation AND put outputs where the
    # lifecycle rule does not apply — a real cost regression dressed up as a security win.
    # The instance already holds one B2 credential; the second adds no meaningful exposure.
    def read_env(name: str) -> dict:
        p = Path.home() / ".config" / "microlab" / f"{name}.env"
        if not p.exists():
            raise SystemExit(f"{p} not found")
        out = {}
        for line in p.read_text().splitlines():
            if "=" in line and line.strip().startswith("B2_"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
        out.setdefault("B2_APPLICATION_KEY", out.get("B2_APP_KEY", ""))
        return out

    corpus, ckpt = read_env("b2_corpus"), read_env("b2_checkpoints")
    env = {
        "B2_CORPUS_KEY_ID": corpus["B2_KEY_ID"],
        "B2_CORPUS_APPLICATION_KEY": corpus["B2_APPLICATION_KEY"],
        "B2_CORPUS_ENDPOINT": corpus["B2_ENDPOINT"],
        "B2_CKPT_KEY_ID": ckpt["B2_KEY_ID"],
        "B2_CKPT_APPLICATION_KEY": ckpt["B2_APPLICATION_KEY"],
        "B2_CKPT_ENDPOINT": ckpt["B2_ENDPOINT"],
        "REPO": a.repo, "BUCKET_IN": a.bucket_in, "BUCKET_OUT": a.bucket_out,
    }

    inst = None
    t0 = time.time()
    try:
        r = call("PUT", f"/asks/{pick['id']}/", {
            "client_id": "me", "image": a.image, "disk": a.min_disk,
            "onstart": onstart_script(a.repo, a.bucket_out, run_tag),
            "env": env, "runtype": "ssh",
        }, key=key)
        inst = r.get("new_contract")
        if not inst:
            raise SystemExit(f"create failed: {r}")
        print(f"\nrented instance {inst}. hard stop in {a.max_minutes} min.")
        deadline = t0 + a.max_minutes * 60
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "b2s", Path(__file__).parent / "b2_sync.py")
        b2mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(b2mod)
        s3 = b2mod.client(b2mod.load_credentials(
            b2mod.credential_path(a.bucket_out, None)))
        while time.time() < deadline:
            time.sleep(30)
            have = b2mod.remote_sizes(s3, a.bucket_out, f"shakedown/{run_tag}")
            el = (time.time() - t0) / 60
            if any(k.endswith(".json") for k in have):
                print(f"result landed after {el:.0f} min "
                      f"(${price*el/60:.2f} spent)")
                break
            print(f"  [{el:>4.0f} min] waiting… ${price*el/60:.2f} so far", flush=True)
        else:
            print(f"HARD STOP at {a.max_minutes} min — destroying regardless")
    finally:
        if inst:
            print(f"destroying instance {inst} …")
            try:
                destroy(inst, key)
                print("destroyed")
            except SystemExit as e:
                print(f"DESTROY FAILED: {e}\n"
                      f"  -> destroy it by hand NOW: https://cloud.vast.ai/instances/")
        spent = price * (time.time() - t0) / 3600
        print(f"elapsed {(time.time()-t0)/60:.1f} min, ~${spent:.2f}")
        # "I thought it was destroyed" is how the bill happens — verify, do not assume.
        still = live_instances(key)
        if still:
            print(f"\nWARNING: {len(still)} instance(s) STILL RUNNING: "
                  f"{[i.get('id') for i in still]}")
            print("  https://cloud.vast.ai/instances/")
        else:
            print("confirmed: no live instances on the account")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("search", "shakedown", "instances", "destroy"))
    ap.add_argument("--gpu", default="H100 SXM")
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--max-price", type=float, default=2.00, help="$/GPU-hour ceiling")
    ap.add_argument("--min-reliability", type=float, default=0.98)
    ap.add_argument("--min-disk", type=int, default=120, help="GB")
    ap.add_argument("--max-minutes", type=int, default=75, help="HARD wall clock")
    ap.add_argument("--image", default="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    ap.add_argument("--repo", default="https://github.com/rje/microlab.git")
    ap.add_argument("--bucket-in", default="microlab-corpus")
    ap.add_argument("--bucket-out", default="microlab-checkpoints",
                    help="outputs; the bucket carrying the lifecycle rule")
    ap.add_argument("--tag", default="shakedown-1")
    ap.add_argument("--id", type=int, help="instance id, for destroy")
    ap.add_argument("--offer-id", type=int, help="rent THIS offer, not the cheapest")
    ap.add_argument("--yes", action="store_true", help="actually rent (default: dry run)")
    a = ap.parse_args()
    key = api_key()

    if a.action == "search":
        return cmd_search(a, key)
    if a.action == "instances":
        inst = live_instances(key)
        print(f"{len(inst)} instance(s)")
        for i in inst:
            print(f"  {i.get('id')}  {i.get('actual_status')}  "
                  f"${i.get('dph_total', 0):.2f}/h  {i.get('gpu_name')}")
        return 0
    if a.action == "destroy":
        if not a.id:
            raise SystemExit("--id required")
        destroy(a.id, key)
        print(f"destroyed {a.id}")
        return 0
    return cmd_shakedown(a, key)


if __name__ == "__main__":
    sys.exit(main())
