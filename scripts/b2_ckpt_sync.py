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
    a = ap.parse_args()

    run = Path(a.run)
    s3 = b2.client(b2.load_credentials(b2.credential_path(a.bucket, None)))
    print(f"watching {run} -> s3://{a.bucket}/{a.prefix} every {a.interval}s", flush=True)

    while True:
        try:
            remote = b2.remote_sizes(s3, a.bucket, a.prefix)
            local = sorted(run.glob("ckpt_*.pt"), key=step_of)
            for p in local:
                key = f"{a.prefix}/{p.name}"
                size = p.stat().st_size
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
            confirmed = [p for p in local
                         if remote.get(f"{a.prefix}/{p.name}") == p.stat().st_size]
            for p in confirmed[:-a.keep_local] if a.keep_local else []:
                if p == local[-1]:
                    continue
                p.unlink()
                print(f"  pruned local {p.name} (confirmed in B2)", flush=True)
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
