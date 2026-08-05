#!/usr/bin/env python
"""Watch a run directory and mirror checkpoints to B2 while training continues.

    python scripts/b2_ckpt_sync.py --run runs/coder-1b --bucket microlab-checkpoints \\
        --prefix coder-1b --interval 120

Runs alongside training on a rented, PREEMPTIBLE box. Local disk dies with the instance, so
a checkpoint that has not reached B2 does not exist — this is the process that makes the
run survivable.

UPLOADS ARE ASYNCHRONOUS BY CONSTRUCTION: this is a separate process, so a slow 16.6 GB
upload never blocks a training step. On the 4x configuration a checkpoint lands every ~33
minutes and takes ~2 minutes to push, which only works because the two are decoupled.

The uploader NEVER deletes a local checkpoint it has not confirmed remotely, and never
deletes the newest one. Losing the file you were about to resume from, in order to save
disk, would be a self-inflicted preemption.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "b2_sync", Path(__file__).resolve().parent / "b2_sync.py")
b2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b2)


def step_of(p: Path) -> int:
    return int(p.stem.split("_")[1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--bucket", default="microlab-checkpoints")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--keep-local", type=int, default=2,
                    help="local checkpoints to retain AFTER remote confirmation")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--env-prefix", default=None,
                    help="credentials from <PREFIX>_KEY_ID/_APPLICATION_KEY/_ENDPOINT")
    ap.add_argument("--log", action="append", default=None,
                    help="also ship this file to B2 each pass; repeatable")
    ap.add_argument("--remote-keep", type=int, default=0,
                    help="prune REMOTE rolling checkpoints beyond the newest N "
                         "(milestones exempt; 0 disables). Without this the bucket "
                         "accumulates every 50-step checkpoint forever: a 44-checkpoint "
                         "run left 405 GB behind, and the full 40k-step run would leave "
                         "7.4 TB (~$44/mo) of files nothing will ever resume from.")
    ap.add_argument("--milestone-interval", type=int, default=0,
                    help="checkpoints at multiples of this step are PERMANENT and never "
                         "remote-pruned. Must match the trainer's "
                         "ckpt_milestone_interval or the emergence trajectory is lost.")
    a = ap.parse_args()

    run = Path(a.run)
    s3 = b2.client(b2.load_credentials(b2.credential_path(a.bucket, None),
                                   a.env_prefix))
    print(f"watching {run} -> s3://{a.bucket}/{a.prefix} every {a.interval}s", flush=True)

    while True:
        try:
            # LOGS. Without these the only thing visible from outside is "alive" and the
            # B2 checkpoint step — which cannot distinguish "downloading the corpus" from
            # "wedged", and cannot explain a failure after the box is destroyed. On a
            # multi-day paid run that is the difference between diagnosing a problem and
            # re-renting to reproduce it. Shipped every pass, overwriting, so the newest
            # tail is always one `b2_sync ls` away.
            # TensorBoard event files ride along with the logs. Val loss is written ONLY
            # to TB during training, so before this, every eval milestone died with the
            # box on preemption — the run's only quality signal was unrecoverable.
            events = sorted(run.glob("events.out.tfevents.*"))
            for path in [*(a.log or []), *map(str, events)]:
                p = Path(path)
                if not p.exists():
                    continue
                try:
                    # Tail only: a training log grows without bound and the useful part is
                    # always the end.
                    data = p.read_bytes()[-256_000:]
                    s3.put_object(Bucket=a.bucket, Key=f"{a.prefix}/logs/{p.name}",
                                  Body=data)
                except Exception as e:              # noqa: BLE001, PERF203
                    print(f"  log ship failed for {p.name}: {e}", flush=True)

            remote = b2.remote_sizes(s3, a.bucket, a.prefix)
            local = sorted(run.glob("ckpt_*.pt"), key=step_of)
            for p in local:
                key = f"{a.prefix}/{p.name}"
                # The TRAINER also prunes this directory (rank 0, ckpt_keep). A file can
                # vanish between our glob and this stat; that made the whole pass abort
                # mid-loop — and everything after it, including log shipping, never ran.
                # The shipped log froze for 30+ minutes while checkpoints kept flowing.
                try:
                    size = p.stat().st_size
                except FileNotFoundError:
                    continue
                if remote.get(key) == size:
                    continue
                # A checkpoint still being written has a size that changes between reads;
                # uploading it would produce a remote file that torch.load cannot open and
                # that the supervisor would happily try to resume from.
                first = size
                time.sleep(3)
                if p.stat().st_size != first:
                    print(f"  {p.name} still being written, skipping this pass", flush=True)
                    continue
                t0 = time.time()
                s3.upload_file(str(p), a.bucket, key)
                el = time.time() - t0
                print(f"  uploaded {p.name} ({size/1e9:.1f} GB in {el:.0f}s = "
                      f"{size/el/1e6:.0f} MB/s)", flush=True)
                remote[key] = size

            # prune ONLY what is confirmed remote, and never the newest
            def _size(q):
                # Same trainer-prune race as the upload loop, one block lower: the file
                # can vanish between glob and stat. A vanished file is simply not
                # confirmable this pass.
                try:
                    return q.stat().st_size
                except FileNotFoundError:
                    return -1
            confirmed = [p for p in local
                         if remote.get(f"{a.prefix}/{p.name}") == _size(p)]
            for p in confirmed[:-a.keep_local] if a.keep_local else []:
                if p == local[-1]:
                    continue
                p.unlink()
                print(f"  pruned local {p.name} (confirmed in B2)", flush=True)
            # REMOTE prune: rolling checkpoints beyond the newest --remote-keep, with
            # milestones exempt. Same safety order as local pruning — never the newest,
            # and only files whose step parses; anything unexpected is left alone.
            if a.remote_keep > 0:
                rolled = []
                for key in b2.remote_sizes(s3, a.bucket, f"{a.prefix}/ckpt_"):
                    try:
                        n = int(key.rsplit("ckpt_", 1)[1].split(".")[0])
                    except (IndexError, ValueError):
                        continue
                    if a.milestone_interval > 0 and n % a.milestone_interval == 0:
                        continue                      # permanent trajectory record
                    rolled.append((n, key))
                rolled.sort()
                for n, key in rolled[:-a.remote_keep]:
                    s3.delete_object(Bucket=a.bucket, Key=key)
                    print(f"  pruned remote ckpt_{n} (rolling window)", flush=True)

        except Exception as e:                      # noqa: BLE001
            # A transient B2 error must not kill the syncer — if it exits, checkpoints stop
            # reaching durable storage and the next preemption loses everything since the
            # last successful push, silently.
            print(f"  sync error (will retry): {type(e).__name__}: {e}", flush=True)
        if a.once:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
