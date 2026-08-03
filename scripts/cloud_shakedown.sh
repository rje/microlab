#!/usr/bin/env bash
# One-hour, ~$1.50 shakedown on a single rented H100. Answers the three things that cannot
# be measured from the local 48 GB card, before committing to a multi-hundred-dollar run.
#
#   1. Does the 1B at 32k fit in 80 GB WITHOUT gradient checkpointing? Worth ~$80 on the
#      total: 6ND is ~159 GPU-h, 8ND is ~212.
#   2. Real tokens/sec and MFU on an H100 -> turns a projection into a measurement.
#   3. B2 -> instance download rate. The one storage number never observed; it sets how
#      much GPU time every job start burns.
#
# Usage on the instance:
#   export REPO=https://github.com/rje/microlab.git
#   export B2_CORPUS_KEY_ID=... B2_CORPUS_APPLICATION_KEY=... B2_CORPUS_ENDPOINT=...
#   bash cloud_shakedown.sh
#
# TWO key pairs, because the buckets are separate on purpose: this script only READS the
# corpus, so it only ever holds the corpus key. Uploading results is a separate step that
# uses B2_CKPT_* and writes to the checkpoints bucket — the one carrying the
# noncurrent-version lifecycle rule that keeps superseded checkpoints from billing forever.
#
# Deliberately NOT `set -e`: a failing rung should report and let the later rungs run, so
# one hour of rented time yields as many answers as possible rather than stopping at the
# first problem. Each step prints its own PASS/FAIL.

set -uo pipefail
WORK=${WORK:-/workspace}
BUCKET=${BUCKET_IN:-microlab-corpus}
PREFIX=${PREFIX:-mix-v1}
SHARDS=${SHARDS:-3}          # enough to train a few steps; the full 39 GB is not needed
# Corpus lives OUTSIDE the clone, so re-cloning does not re-download 40 GB.
CORPUS=${CORPUS:-$WORK/mix-v1}
mkdir -p "$WORK" && cd "$WORK"

say() { printf '\n=== %s ===\n' "$*"; }

say "0. environment"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,
'bf16',torch.cuda.is_bf16_supported())" 2>/dev/null || echo "FAIL: no torch"

say "1. dependencies"
pip install -q boto3 tokenizers tensorboard 2>&1 | tail -2
# flash-linear-attention supplies the fused KDA kernel. Without it the model still runs on
# the float64 reference path, which is an ORACLE and ~25x slower — any throughput number
# measured that way would be meaningless, so this failing is fatal to the benchmark.
pip install -q flash-linear-attention==0.5.2 2>&1 | tail -2
python -c "from fla.ops.kda import chunk_kda; print('fla OK')" || echo "FAIL: fla missing"

say "2. code from GitHub"
# git, not a tarball in the bucket: the instance then reports the exact commit it ran, so
# a result can be tied to a revision instead of to "whatever was packaged that day".
# REPO must be a clone URL the instance can read — for a private repo use
#   https://<token>@github.com/<owner>/microlab.git
# passed in the environment, never baked into this file.
: "${REPO:?set REPO to the clone URL}"
rm -rf microlab && git clone --depth 1 "$REPO" microlab || echo "FAIL: clone"
cd microlab || exit 1
echo "commit: $(git rev-parse --short HEAD)  $(git log -1 --format=%s)"

say "3. corpus download rate (the unmeasured storage number)"
python - <<PY
import boto3, os, json, time
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=os.environ["B2_CORPUS_ENDPOINT"],
                  aws_access_key_id=os.environ["B2_CORPUS_KEY_ID"],
                  aws_secret_access_key=os.environ["B2_CORPUS_APPLICATION_KEY"],
                  config=Config(retries={"max_attempts":10,"mode":"adaptive"},
                                max_pool_connections=32))
B, P, N = "$BUCKET", "$CORPUS", $SHARDS
os.makedirs(P, exist_ok=True)
cfg = TransferConfig(multipart_threshold=200*1024**2, multipart_chunksize=200*1024**2,
                     max_concurrency=32)
for f in ("train-manifest.json","val-manifest.json","tokenizer.json"):
    s3.download_file(B, f"$PREFIX/{f}", f"{P}/{f}")
man = json.load(open(f"{P}/train-manifest.json"))
keep = man["shards"][:N]
t0 = time.time(); total = 0
for s in keep:
    s3.download_file(B, f"$PREFIX/{s['file']}", f"{P}/{s['file']}", Config=cfg)
    total += os.path.getsize(f"{P}/{s['file']}")
el = time.time()-t0
print(f"downloaded {total/1e9:.1f} GB in {el:.0f}s = {total/el/1e6:.0f} MB/s")
print(f"  -> full 39.2 GB corpus would take {39.2e9/(total/el)/60:.1f} min")
# trimmed manifest so ShardDataset does not look for shards we did not fetch
json.dump({"split":"train","dtype":"uint16","shards":keep,
           "total_tokens":sum(s["tokens"] for s in keep)},
          open(f"{P}/train-manifest.json","w"))
vman = json.load(open(f"{P}/val-manifest.json"))
s3.download_file(B, f"$PREFIX/{vman['shards'][0]['file']}", f"{P}/{vman['shards'][0]['file']}")
json.dump({"split":"val","dtype":"uint16","shards":vman["shards"][:1],
           "total_tokens":vman["shards"][0]["tokens"]}, open(f"{P}/val-manifest.json","w"))
PY

say "4. THE question: does 1B@32k fit in 80GB without gradient checkpointing?"
sed -i "s#data_dir=\"data/shards/mix-v1\"#data_dir=\"$CORPUS\"#" configs/coder-1b.py
# ckpt=1 is the local baseline (known to fit); ckpt=0 rows are what 80 GB might unlock.
python -u scripts/bench_train_config.py --config configs/coder-1b.py --steps 6 \
  --variants "32768,1,1,0;32768,1,0,0;32768,2,0,0;32768,4,0,0" \
  --out shakedown-bench.json

say "5. verdict"
python - <<'PY'
import json
N, H100 = 1.013e9, 494.7e12
try: r = json.load(open("shakedown-bench.json"))["results"]
except Exception: raise SystemExit("no bench output — step 4 failed")
ok = [x for x in r if x.get("status") == "ok"]
if not ok: raise SystemExit("every variant failed")
print(f"{'block':>7} {'bs':>3} {'ckpt':>5} {'tok/s':>9} {'peak GB':>8} {'MFU':>6} {'GPU-h':>7} {'$@1.49':>8}")
for x in sorted(ok, key=lambda y: -y["tok_s"]):
    mult = 8 if x["grad_checkpoint"] else 6
    mfu = mult*N*x["tok_s"]/H100
    gh = 21.0e9/x["tok_s"]/3600
    print(f"{x['block']:>7,} {x['batch_size']:>3} {str(x['grad_checkpoint']):>5} "
          f"{x['tok_s']:>9,.0f} {x['peak_gb']:>8.1f} {mfu:>5.0%} {gh:>7.0f} {gh*1.49:>8.0f}")
best = max(ok, key=lambda y: y["tok_s"])
gh = 21.0e9/best["tok_s"]/3600
print(f"\nBEST: {best['tok_s']:,.0f} tok/s -> {gh:.0f} GPU-h = {gh/24:.1f} days on ONE H100")
print(f"  at $1.49/h = ${gh*1.49:,.0f}   at $1.74/h = ${gh*1.74:,.0f}")
print(f"  local for comparison: 46 days, ~$180-200 of PG&E EV2-A electricity")
print("\nGATE: proceed only if MFU >= 30% and a no-checkpointing row actually ran.")
PY
