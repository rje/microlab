#!/usr/bin/env python
"""Sync the corpus and checkpoints to/from Backblaze B2 (S3-compatible).

    python scripts/b2_sync.py up   --local data/shards/mix-v1 --bucket microlab-corpus \\
        --prefix mix-v1
    python scripts/b2_sync.py down --local /workspace/mix-v1  --bucket microlab-corpus \\
        --prefix mix-v1
    python scripts/b2_sync.py verify --local data/shards/mix-v1 --bucket microlab-corpus \\
        --prefix mix-v1

B2 rather than R2: Vast.ai's cloud-sync supports S3, Backblaze, Google Drive and Dropbox
but NOT R2, and B2 is also cheaper at our shape ($1.53/mo vs $3.15/mo for 220 GB). R2's
one advantage was uncapped free egress, but B2's cap is 3x stored — 660 GB/month for us
against ~216 GB of actual use, so it does not bind.

CREDENTIALS are read from ~/.config/microlab/b2.env (mode 600), never from the repo and
never from the command line (argv is world-readable via /proc). Expected keys:

    B2_KEY_ID=...
    B2_APP_KEY=...
    B2_ENDPOINT=https://s3.us-west-004.backblazeb2.com   # from the B2 bucket page

INTEGRITY IS CHECKED, NOT ASSUMED. A truncated shard would not raise — it would train on
short data and produce a subtly wrong model. `verify` compares every object's size against
the local file and re-checks the token totals in the manifests, so "the upload finished"
and "the corpus is intact" are different claims with different evidence.

Uploads are resumable: an object whose remote size already matches is skipped, so an
interrupted 40 GB upload continues rather than restarting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

CRED_DIR = Path.home() / ".config" / "microlab"


def credential_path(bucket: str, explicit: str | None) -> Path:
    """Which env file holds this bucket's key.

    Keys are scoped PER BUCKET (a B2 application key covers one bucket or all of them, so
    one key per bucket is the least-privilege arrangement). The file is therefore chosen
    from the bucket name — `microlab-corpus` -> `b2_corpus.env` — with `b2.env` as the
    single-bucket fallback.
    """
    if explicit:
        return Path(explicit).expanduser()
    suffix = bucket.rsplit("-", 1)[-1]
    for cand in (CRED_DIR / f"b2_{suffix}.env", CRED_DIR / f"b2_{bucket}.env",
                 CRED_DIR / "b2.env"):
        if cand.exists():
            return cand
    return CRED_DIR / f"b2_{suffix}.env"


def load_credentials(path: Path, env_prefix: str | None = None) -> dict:
    """Read B2 creds from `path`, falling back to the process environment.

    `env_prefix` names a PREFIXED set of environment variables — B2_CKPT_KEY_ID and
    friends — which is how a rented instance receives them: there is no env FILE on that
    box, only variables, and they must be prefixed because the run holds two separately
    scoped keys (corpus read, checkpoints write).

    This is the bug that cost a real run. The checkpoint syncer looked only for an env
    file and then for UNPREFIXED variables, found neither, and failed on every pass — so
    two 9.2 GB checkpoints sat on disk that dies with the instance, the supervisor's only
    progress signal never arrived, and it destroyed a run that had actually finished.
    """
    creds: dict[str, str] = {}
    if env_prefix:
        got = {k: os.environ.get(f"{env_prefix}_{k}", "")
               for k in ("KEY_ID", "APPLICATION_KEY", "ENDPOINT")}
        if all(got.values()):
            return {"B2_KEY_ID": got["KEY_ID"],
                    "B2_APP_KEY": got["APPLICATION_KEY"],
                    "B2_ENDPOINT": got["ENDPOINT"]}
    if path.exists():
        mode = path.stat().st_mode & 0o777
        if mode & 0o077:
            raise SystemExit(
                f"{path} is mode {mode:o}; it holds a secret. Run: chmod 600 {path}")
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            creds[k.strip()] = v.strip().strip('"').strip("'")
    # B2's console labels the secret "applicationKey"; accept either spelling rather than
    # failing on a name the user reasonably copied from the UI.
    key = creds.get("B2_APP_KEY") or creds.get("B2_APPLICATION_KEY") or \
        os.environ.get("B2_APP_KEY", "") or os.environ.get("B2_APPLICATION_KEY", "")
    out = {
        "B2_KEY_ID": creds.get("B2_KEY_ID") or os.environ.get("B2_KEY_ID", ""),
        "B2_APP_KEY": key,
        "B2_ENDPOINT": creds.get("B2_ENDPOINT") or os.environ.get("B2_ENDPOINT", ""),
    }
    missing = [k for k, v in out.items() if not v]
    if missing:
        raise SystemExit(
            f"missing {missing} in {path}. Expected (chmod 600):\n"
            f"  B2_KEY_ID=...\n  B2_APPLICATION_KEY=...\n"
            f"  B2_ENDPOINT=https://s3.<region>.backblazeb2.com")
    return out


def client(creds: dict):
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=creds["B2_ENDPOINT"],
        aws_access_key_id=creds["B2_KEY_ID"],
        aws_secret_access_key=creds["B2_APP_KEY"],
        # B2 throttles aggressively on burst; adaptive retries back off rather than
        # hammering and failing a 40 GB upload three quarters of the way through.
        config=_b2_config(),
    )


def _b2_config():
    from botocore.config import Config
    base = dict(retries={"max_attempts": 10, "mode": "adaptive"},
                max_pool_connections=16)
    # botocore >= 1.36 computes streaming (aws-chunked) checksums BY DEFAULT, and a
    # retried upload then dies with "UnseekableStreamError: Need to rewind the stream" —
    # against B2, every transient reset became a permanently failed pass, and checkpoint
    # uploads fell 330 steps behind training on a transpacific link. "when_required"
    # restores the old behavior; older botocore rejects the kwarg, hence the fallback.
    try:
        return Config(request_checksum_calculation="when_required", **base)
    except TypeError:
        return Config(**base)


def local_files(local: Path) -> list[Path]:
    return sorted(p for p in local.rglob("*") if p.is_file())


def remote_sizes(s3, bucket: str, prefix: str) -> dict[str, int]:
    out, token = {}, None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            out[o["Key"]] = o["Size"]
        if not resp.get("IsTruncated"):
            return out
        token = resp["NextContinuationToken"]


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def cmd_up(a, s3) -> int:
    local = Path(a.local)
    files = local_files(local)
    have = remote_sizes(s3, a.bucket, a.prefix)
    total = sum(f.stat().st_size for f in files)
    todo = [f for f in files
            if have.get(f"{a.prefix}/{f.relative_to(local)}") != f.stat().st_size]
    print(f"{len(files)} files, {human(total)} total; {len(todo)} need upload "
          f"({human(sum(f.stat().st_size for f in todo))})")
    from boto3.s3.transfer import TransferConfig
    cfg = TransferConfig(multipart_threshold=200 * 1024**2,
                         multipart_chunksize=200 * 1024**2, max_concurrency=8)
    sent, t0 = 0, time.time()
    want = sum(x.stat().st_size for x in todo)
    for i, f in enumerate(todo, 1):
        key = f"{a.prefix}/{f.relative_to(local)}"
        s3.upload_file(str(f), a.bucket, key, Config=cfg)
        sent += f.stat().st_size
        el = time.time() - t0
        print(f"  [{i}/{len(todo)}] {key}  {human(sent)}/{human(want)}"
              f"  {human(sent/max(el, 1))}/s", flush=True)
    print(f"done in {time.time()-t0:.0f}s")
    return cmd_verify(a, s3)


def cmd_down(a, s3) -> int:
    local = Path(a.local)
    local.mkdir(parents=True, exist_ok=True)
    have = remote_sizes(s3, a.bucket, a.prefix)
    if not have:
        raise SystemExit(f"nothing under s3://{a.bucket}/{a.prefix} — wrong prefix?")
    total = sum(have.values())
    print(f"{len(have)} objects, {human(total)}")
    from boto3.s3.transfer import TransferConfig
    cfg = TransferConfig(multipart_threshold=200 * 1024**2,
                         multipart_chunksize=200 * 1024**2, max_concurrency=16)
    got, t0 = 0, time.time()
    for i, (key, size) in enumerate(sorted(have.items()), 1):
        dest = local / key[len(a.prefix) + 1:]
        if dest.exists() and dest.stat().st_size == size:
            got += size
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(a.bucket, key, str(dest), Config=cfg)
        got += size
        el = time.time() - t0
        print(f"  [{i}/{len(have)}] {dest.name}  {human(got)}/{human(total)}"
              f"  {human(got/max(el,1))}/s", flush=True)
    el = time.time() - t0
    print(f"done in {el:.0f}s ({human(total/max(el,1))}/s)")
    return cmd_verify(a, s3)


def cmd_verify(a, s3) -> int:
    """Sizes must match object-for-object, AND the manifests must still add up."""
    local = Path(a.local)
    have = remote_sizes(s3, a.bucket, a.prefix)
    problems = []
    for f in local_files(local):
        key = f"{a.prefix}/{f.relative_to(local)}"
        if key not in have:
            problems.append(f"missing remotely: {key}")
        elif have[key] != f.stat().st_size:
            problems.append(f"size mismatch {key}: local {f.stat().st_size:,} "
                            f"remote {have[key]:,}")
    # A shard truncated in transit would still be a valid .bin and would train fine on
    # short data, so check the manifests' own arithmetic against what is on disk.
    for split in ("train", "val"):
        man = local / f"{split}-manifest.json"
        if not man.exists():
            continue
        m = json.loads(man.read_text())
        declared = m["total_tokens"]
        actual = 0
        for s in m["shards"]:
            p = local / s["file"]
            if not p.exists():
                problems.append(f"{split}: manifest lists missing shard {s['file']}")
                continue
            n = p.stat().st_size // 2          # uint16
            if n != s["tokens"]:
                problems.append(f"{split}/{s['file']}: manifest says {s['tokens']:,} "
                                f"tokens, file holds {n:,}")
            actual += n
        if actual != declared:
            problems.append(f"{split}: manifest total {declared:,} != sum of shards "
                            f"{actual:,}")
        else:
            print(f"  {split}-manifest: {declared:,} tokens across {len(m['shards'])} "
                  f"shards — verified")
    if problems:
        print("\nVERIFY FAILED:")
        for p in problems[:20]:
            print(f"  {p}")
        return 1
    print(f"verified {len(local_files(local))} files against s3://{a.bucket}/{a.prefix}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("up", "down", "verify", "ls"))
    ap.add_argument("--local", required=False)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--creds", default=None,
                    help="env file with this bucket's key; defaults to "
                         "~/.config/microlab/b2_<bucket suffix>.env")
    ap.add_argument("--env-prefix", default=None,
                    help="read credentials from <PREFIX>_KEY_ID/_APPLICATION_KEY/_ENDPOINT "
                         "instead of a file. Rented instances have no env file — only "
                         "prefixed variables, because the run carries two scoped keys.")
    a = ap.parse_args()

    cred_path = credential_path(a.bucket, a.creds)
    src = f"env {a.env_prefix}_*" if a.env_prefix else str(cred_path)
    print(f"bucket {a.bucket}  creds {src}")
    s3 = client(load_credentials(cred_path, a.env_prefix))
    if a.action == "ls":
        have = remote_sizes(s3, a.bucket, a.prefix)
        for k, v in sorted(have.items()):
            print(f"{v:>14,}  {k}")
        print(f"{len(have)} objects, {human(sum(have.values()))}")
        return 0
    if not a.local:
        raise SystemExit("--local is required for up/down/verify")
    return {"up": cmd_up, "down": cmd_down, "verify": cmd_verify}[a.action](a, s3)


if __name__ == "__main__":
    sys.exit(main())
