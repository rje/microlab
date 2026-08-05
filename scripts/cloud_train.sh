#!/usr/bin/env bash
# Instance-side entrypoint for a preemptible training run. Idempotent: this is what runs on
# EVERY instance, whether it is the first one or the fifth after four preemptions.
#
# Required environment (supplied by scripts/vast_supervisor.py):
#   REPO NGPU CONFIG RUN_PREFIX
#   B2_CORPUS_KEY_ID B2_CORPUS_APPLICATION_KEY B2_CORPUS_ENDPOINT
#   B2_CKPT_KEY_ID   B2_CKPT_APPLICATION_KEY   B2_CKPT_ENDPOINT
#
# The two credential pairs are separate on purpose: the corpus is read-only to this box and
# checkpoints go to the bucket carrying the noncurrent-version lifecycle rule.

set -uo pipefail
WORK=${WORK:-/workspace}
export WORK
SHARD_PREFIX=${SHARD_PREFIX:-mix-v1}
CORPUS=${CORPUS:-$WORK/$SHARD_PREFIX}
RUNDIR=${RUNDIR:-$WORK/run}
NGPU=${NGPU:-1}
CONFIG=${CONFIG:-configs/coder-1b.py}
BUCKET_IN=${BUCKET_IN:-microlab-corpus}
BUCKET_OUT=${BUCKET_OUT:-microlab-checkpoints}
mkdir -p "$WORK" && cd "$WORK"
say() { printf '\n=== %s ===\n' "$*"; }

say "environment"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch;print('torch',torch.__version__,'gpus',torch.cuda.device_count())"

say "early log shipping"
# Setup on a slow-pipe host is 30-45 minutes, and until this existed NOTHING reached B2
# before the deps phase finished — from outside, "downloading torch at 11 MB/s" and
# "dead" were the same picture, on a box billing four GPUs. A minimal shipper starts
# within the first minute (boto3 alone installs in seconds) and pushes the log every 45 s
# until the real syncer takes over, so every phase marker below is visible as it happens.
pip install -q boto3 2>&1 | tail -1
python - <<'PYEARLY' &
import boto3, os, time
from botocore.config import Config
s3 = boto3.client('s3', endpoint_url=os.environ['B2_CKPT_ENDPOINT'],
    aws_access_key_id=os.environ['B2_CKPT_KEY_ID'],
    aws_secret_access_key=os.environ['B2_CKPT_APPLICATION_KEY'],
    config=Config(retries={'max_attempts': 4, 'mode': 'adaptive'}))
work = os.environ.get('WORK', '/workspace')
while not os.path.exists(f'{work}/.syncer-started'):
    try:
        with open(f'{work}/train.log', 'rb') as f:
            s3.put_object(Bucket=os.environ.get('BUCKET_OUT', 'microlab-checkpoints'),
                          Key=f"{os.environ['RUN_PREFIX']}/logs/train.log",
                          Body=f.read()[-256_000:])
    except Exception:
        pass
    time.sleep(45)
PYEARLY
EARLY_SHIP_PID=$!
echo "early shipper pid $EARLY_SHIP_PID"

say "dependencies"
# torchvision/torchaudio pin an older torch and break the fla import chain with
# "operator torchvision::nms does not exist". They are unused here.
pip uninstall -y -q torchvision torchaudio 2>/dev/null
# Pin the EXACT torch validated locally: 2.12.1. Two independent reasons, both measured:
#   * fla 0.5.2 needs Triton >= 3.6; stock PyTorch 2.5 images ship 3.1.0 and die with
#     "Autotuner.__init__() got an unexpected keyword argument 'do_bench'".
#   * torch 2.11 breaks torch.compile + Liger fused cross-entropy —
#       TypeError: unsupported operand type(s) for *: 'torch.dtype' and 'FakeTensor'
#     which killed a paid run. 2.12.1 does not: configs/frontier-32k.py ran 15,000 steps
#     locally with compile AND fused CE both on.
# Index choice matters and is why we hit the bug: cu128 TOPS OUT at 2.11.0, so
# "--upgrade torch --index-url .../cu128" silently pinned the broken version. cu126 carries
# 2.12.1, and CUDA 12.x minor-version compatibility runs it on driver 550+; cu130 would
# need 580+, which rented boxes do not reliably have.
python - <<'PY'
import subprocess, sys
import torch
if not torch.__version__.startswith("2.12"):
    print(f"torch {torch.__version__} -> installing 2.12.1+cu126 (the validated version)")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.12.1",
                    "--index-url", "https://download.pytorch.org/whl/cu126"], check=False)
PY
python -c "
import torch, triton
assert torch.__version__.startswith('2.12'), f'need torch 2.12, got {torch.__version__}'
assert tuple(int(x) for x in triton.__version__.split('.')[:2]) >= (3, 6), triton.__version__
print('torch', torch.__version__, 'triton', triton.__version__)" \
  || { echo 'FATAL: wrong torch/triton — compile+fused-CE would break'; exit 1; }
pip install -q flash-linear-attention==0.5.2 liger-kernel boto3 tokenizers tensorboard 2>&1 | tail -1
python -c "from fla.ops.kda import chunk_kda; import liger_kernel; print('kernels OK')" \
  || { echo "FATAL: fused kernels unavailable — refusing to train on the float64 oracle path"; exit 1; }

say "code"
: "${REPO:?}"
# The inner clone was the one UNGUARDED network call in this script, and it found the
# way to hurt: on a flaky transpacific pipe it failed right after `rm -rf microlab`
# removed the outer clone — and bash kept executing this (deleted) script from its open
# fd, so every heredoc phase below ran normally and the failure only surfaced at the
# train phase as "cd: /workspace/microlab: No such file". Retry, then die LOUDLY.
rm -rf microlab
for attempt in 1 2 3; do
  git clone -q --depth 1 "$REPO" microlab && break
  echo "clone attempt $attempt failed; retrying in 20s"
  sleep 20
done
[ -d microlab ] || { echo "MICROLAB_TRAIN_FAILED rc=95 (git clone failed 3x)"; exit 95; }
cd microlab
export PYTHONPATH=$WORK/microlab/src
echo "commit $(git rev-parse --short HEAD)"

# Ship the log once now: setup (deps, corpus pull) can take 20+ minutes, and without
# this the run is a black box during exactly the window where things go wrong.
python -u scripts/b2_ckpt_sync.py --run "$RUNDIR" --bucket "$BUCKET_OUT" \
  --prefix "$RUN_PREFIX" --once --keep-local 0 --env-prefix B2_CKPT \
  --log "$WORK/train.log" 2>/dev/null || true

say "corpus: manifests only — shards stream on demand"
# NOT a 39 GB pull. Training reads shards in random order and touches one per sequence, so
# only the manifests and tokenizer are needed before step 1. Measured full-pull cost was
# 9 min from US-West and 54 min from Asia, paid again on EVERY re-provision — and it was
# the only reason host choice had to be constrained by distance to the bucket.
python - <<PY
import boto3, os
from botocore.config import Config
s3 = boto3.client('s3', endpoint_url=os.environ['B2_CORPUS_ENDPOINT'],
    aws_access_key_id=os.environ['B2_CORPUS_KEY_ID'],
    aws_secret_access_key=os.environ['B2_CORPUS_APPLICATION_KEY'],
    config=Config(retries={'max_attempts':10,'mode':'adaptive'}))
P='$CORPUS'; os.makedirs(P, exist_ok=True)
for f in ('train-manifest.json','val-manifest.json','tokenizer.json'):
    s3.download_file('$BUCKET_IN', f'$SHARD_PREFIX/{f}', f'{P}/{f}')
print('manifests + tokenizer ready; shards stream during training')
PY
export MICROLAB_SHARD_BUCKET="$BUCKET_IN"
export MICROLAB_SHARD_PREFIX="$SHARD_PREFIX"

say "compile caches: restore from B2 if present"
# Inductor + Triton caches are keyed by kernel source and tuning shape, and every host we
# rent is sm90 — kernels autotuned on one H100 are valid on the next. Without this, every
# re-provision re-pays 15-20 minutes of max-autotune across 24 blocks at full 4-GPU
# price; with it, compile is ~2-3 minutes of cache validation.
export TORCHINDUCTOR_CACHE_DIR=$WORK/inductor-cache
export TRITON_CACHE_DIR=$WORK/triton-cache
python - <<PYCACHE
import boto3, io, os, tarfile
from botocore.config import Config
s3 = boto3.client('s3', endpoint_url=os.environ['B2_CKPT_ENDPOINT'],
    aws_access_key_id=os.environ['B2_CKPT_KEY_ID'],
    aws_secret_access_key=os.environ['B2_CKPT_APPLICATION_KEY'],
    config=Config(retries={'max_attempts':6,'mode':'adaptive'}))
try:
    obj = s3.get_object(Bucket='$BUCKET_OUT', Key='$RUN_PREFIX/caches/compile-cache.tar')
    tarfile.open(fileobj=io.BytesIO(obj['Body'].read())).extractall('$WORK')
    print('compile cache restored')
except Exception as e:
    print(f'no compile cache yet ({type(e).__name__}) — this episode pays full autotune')
PYCACHE

say "resume: newest checkpoint from B2, if any"
mkdir -p "$RUNDIR"
python - <<PY
import boto3, os
from botocore.config import Config
s3 = boto3.client('s3', endpoint_url=os.environ['B2_CKPT_ENDPOINT'],
    aws_access_key_id=os.environ['B2_CKPT_KEY_ID'],
    aws_secret_access_key=os.environ['B2_CKPT_APPLICATION_KEY'],
    config=Config(retries={'max_attempts':10,'mode':'adaptive'}, max_pool_connections=16))
r = s3.list_objects_v2(Bucket='$BUCKET_OUT', Prefix='$RUN_PREFIX/ckpt_')
objs = r.get('Contents', [])
if not objs:
    print('no remote checkpoint — starting from step 0')
else:
    # Resume from the HIGHEST step, not the most recently modified: a re-upload of an older
    # checkpoint would otherwise rewind the run without anything noticing.
    newest = max(objs, key=lambda o: int(o['Key'].rsplit('ckpt_',1)[1].split('.')[0]))
    dest = os.path.join('$RUNDIR', os.path.basename(newest['Key']))
    print(f"resuming from {newest['Key']} ({newest['Size']/1e9:.1f} GB)", flush=True)
    import time as _t
    state = {"done": 0, "mark": 0, "t0": _t.time()}
    def cb(n):
        state["done"] += n
        pct = 100 * state["done"] / newest['Size']
        if pct - state["mark"] >= 10:
            state["mark"] = pct
            rate = state["done"] / max(_t.time() - state["t0"], 1) / 1e6
            print(f"  [resume] {pct:.0f}% {state['done']/1e9:.1f}/"
                  f"{newest['Size']/1e9:.1f} GB @ {rate:.0f} MB/s", flush=True)
    s3.download_file('$BUCKET_OUT', newest['Key'], dest, Callback=cb)
    assert os.path.getsize(dest) == newest['Size'], 'checkpoint truncated in transit'
PY
# The tokenizer must sit beside the checkpoints for the run dir to be self-describing.
cp "$CORPUS/tokenizer.json" "$RUNDIR/" 2>/dev/null || true

say "checkpoint syncer (background)"
python -u scripts/b2_ckpt_sync.py --run "$RUNDIR" --bucket "$BUCKET_OUT" \
  --prefix "$RUN_PREFIX" --interval 120 --env-prefix B2_CKPT \
  --remote-keep 3 --milestone-interval 2000 \
  --log "$WORK/train.log" --log "$WORK/ckptsync.log" > "$WORK/ckptsync.log" 2>&1 &
SYNC_PID=$!
echo "syncer pid $SYNC_PID"
touch "$WORK/.syncer-started"    # retires the early log shipper

# Ship the compile cache ~25 min in — after autotune has finished — in the background,
# so even an episode that later gets preempted usually leaves the next one a warm cache.
# Skipped when the tar would be empty (compile off) because upload_file would fail late.
( sleep 1500
  if tar -C "$WORK" -cf "$WORK/compile-cache.tar" inductor-cache triton-cache 2>/dev/null; then
    python - <<PYSHIP
import boto3, os
from botocore.config import Config
s3 = boto3.client('s3', endpoint_url=os.environ['B2_CKPT_ENDPOINT'],
    aws_access_key_id=os.environ['B2_CKPT_KEY_ID'],
    aws_secret_access_key=os.environ['B2_CKPT_APPLICATION_KEY'],
    config=Config(retries={'max_attempts':6,'mode':'adaptive'}))
s3.upload_file('$WORK/compile-cache.tar', '$BUCKET_OUT',
               '$RUN_PREFIX/caches/compile-cache.tar')
print('compile cache shipped to B2', flush=True)
PYSHIP
  fi
) &

say "train: $NGPU GPU(s)"
# Belt and braces: something in the setup phases left cwd off the repo once — the
# config verifier then resolved its RELATIVE path from /workspace, crashed uncaught,
# and the script exited without the failure sentinel: a silently idle box billing four
# GPUs. Pin cwd immediately before anything touches the config.
cd "$WORK/microlab" || { echo "MICROLAB_TRAIN_FAILED rc=96 (repo dir missing)"; exit 96; }
# Rewrite by PATTERN, then PROVE the rewrite took. sed exits 0 on a no-match, and an
# unmatched literal here once meant the trainer wrote to a path the checkpoint syncer was
# not watching — a full rental of nothing durable, invisible to the watchdog because the
# log keeps shipping. The assert imports the config the same way the trainer does.
sed -i "s#data_dir=\"data/shards/[^\"]*\"#data_dir=\"$CORPUS\"#; s#out_dir=\"runs/[^\"]*\"#out_dir=\"$RUNDIR\"#" "$CONFIG"
python - <<PYCHK
import importlib.util, os, sys
try:
    spec = importlib.util.spec_from_file_location("run_config", os.path.abspath("$CONFIG"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
except Exception as e:  # the VERIFIER failing must be as loud as what it verifies
    print(f"FATAL: could not load config $CONFIG from {os.getcwd()}: {e}")
    print("MICROLAB_TRAIN_FAILED rc=97")
    sys.exit(97)
problems = []
if os.path.abspath(m.config.data_dir) != os.path.abspath("$CORPUS"):
    problems.append(f"data_dir={m.config.data_dir!r} != $CORPUS")
if os.path.abspath(m.config.out_dir) != os.path.abspath("$RUNDIR"):
    problems.append(f"out_dir={m.config.out_dir!r} != $RUNDIR")
if problems:
    print("FATAL: config rewrite did not take:", "; ".join(problems))
    print("MICROLAB_TRAIN_FAILED rc=97")
    sys.exit(97)
print("config paths verified:", m.config.data_dir, m.config.out_dir)
PYCHK
[ $? -eq 0 ] || exit 97
torchrun --nproc_per_node="$NGPU" scripts/pretrain.py "$CONFIG"
RC=$?
# EXPLICIT failure sentinel. The supervisor must not have to infer a crash from log text:
# a heuristic scan for "Error" matched the benign "ModuleNotFoundError: nvidia-ml-py"
# telemetry notice and destroyed a healthy box. We control both ends of this channel, so
# the signal is a literal string that appears only on a non-zero exit.
if [ "$RC" -ne 0 ]; then
  echo "MICROLAB_TRAIN_FAILED rc=$RC"
fi

say "final flush"
# One synchronous pass so the last checkpoint reaches B2 before this box goes away.
kill $SYNC_PID 2>/dev/null
python -u scripts/b2_ckpt_sync.py --run "$RUNDIR" --bucket "$BUCKET_OUT" \
  --prefix "$RUN_PREFIX" --once --keep-local 0 --env-prefix B2_CKPT \
  --log "$WORK/train.log"
echo "exit $RC"
exit $RC
