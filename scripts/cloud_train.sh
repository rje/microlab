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
CORPUS=${CORPUS:-$WORK/mix-v1}
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

say "dependencies"
# torchvision/torchaudio pin an older torch and break the fla import chain with
# "operator torchvision::nms does not exist". They are unused here.
pip uninstall -y -q torchvision torchaudio 2>/dev/null
# fla 0.5.2 needs Triton >= 3.6; the stock PyTorch 2.5 images ship 3.1.0 and fail with
# "Autotuner.__init__() got an unexpected keyword argument 'do_bench'". Measured on a real
# instance, not assumed.
python - <<'PY'
import subprocess, sys
import triton
if tuple(int(x) for x in triton.__version__.split(".")[:2]) < (3, 6):
    print(f"triton {triton.__version__} too old for fla; upgrading torch")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "torch",
                    "--index-url", "https://download.pytorch.org/whl/cu128"], check=False)
PY
pip install -q flash-linear-attention==0.5.2 liger-kernel boto3 tokenizers tensorboard 2>&1 | tail -1
python -c "from fla.ops.kda import chunk_kda; import liger_kernel; print('kernels OK')" \
  || { echo "FATAL: fused kernels unavailable — refusing to train on the float64 oracle path"; exit 1; }

say "code"
: "${REPO:?}"
rm -rf microlab && git clone -q --depth 1 "$REPO" microlab
cd microlab
export PYTHONPATH=$WORK/microlab/src
echo "commit $(git rev-parse --short HEAD)"

say "corpus"
python - <<PY
import boto3, os, json, time
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
s3 = boto3.client('s3', endpoint_url=os.environ['B2_CORPUS_ENDPOINT'],
    aws_access_key_id=os.environ['B2_CORPUS_KEY_ID'],
    aws_secret_access_key=os.environ['B2_CORPUS_APPLICATION_KEY'],
    config=Config(retries={'max_attempts':10,'mode':'adaptive'}, max_pool_connections=32))
P='$CORPUS'; os.makedirs(P, exist_ok=True)
cfg=TransferConfig(multipart_threshold=200*1024**2, multipart_chunksize=200*1024**2,
                   max_concurrency=32)
tok=0; t0=time.time()
token=None; keys={}
while True:
    kw={'Bucket':'$BUCKET_IN','Prefix':'mix-v1'}
    if token: kw['ContinuationToken']=token
    r=s3.list_objects_v2(**kw)
    for o in r.get('Contents',[]): keys[o['Key']]=o['Size']
    if not r.get('IsTruncated'): break
    token=r['NextContinuationToken']
for k,size in sorted(keys.items()):
    dest=os.path.join(P, k.split('/',1)[1])
    if os.path.exists(dest) and os.path.getsize(dest)==size:
        tok+=size; continue          # resumable: a re-provision re-uses what survived
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    s3.download_file('$BUCKET_IN', k, dest, Config=cfg)
    tok+=size
el=time.time()-t0
print(f'corpus {tok/1e9:.1f} GB in {el:.0f}s = {tok/max(el,1)/1e6:.0f} MB/s')
PY

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
    print(f"resuming from {newest['Key']} ({newest['Size']/1e9:.1f} GB)")
    s3.download_file('$BUCKET_OUT', newest['Key'], dest)
    assert os.path.getsize(dest) == newest['Size'], 'checkpoint truncated in transit'
PY
# The tokenizer must sit beside the checkpoints for the run dir to be self-describing.
cp "$CORPUS/tokenizer.json" "$RUNDIR/" 2>/dev/null || true

say "checkpoint syncer (background)"
python -u scripts/b2_ckpt_sync.py --run "$RUNDIR" --bucket "$BUCKET_OUT" \
  --prefix "$RUN_PREFIX" --interval 120 > "$WORK/ckptsync.log" 2>&1 &
SYNC_PID=$!
echo "syncer pid $SYNC_PID"

say "train: $NGPU GPU(s)"
sed -i "s#data_dir=\"data/shards/mix-v1\"#data_dir=\"$CORPUS\"#; s#out_dir=\"runs/coder-1b\"#out_dir=\"$RUNDIR\"#" "$CONFIG"
torchrun --nproc_per_node="$NGPU" scripts/pretrain.py "$CONFIG"
RC=$?

say "final flush"
# One synchronous pass so the last checkpoint reaches B2 before this box goes away.
kill $SYNC_PID 2>/dev/null
python -u scripts/b2_ckpt_sync.py --run "$RUNDIR" --bucket "$BUCKET_OUT" \
  --prefix "$RUN_PREFIX" --once --keep-local 0
echo "exit $RC"
exit $RC
