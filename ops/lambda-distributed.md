# Phase 7 cloud drills — Lambda (or equivalent) runbook

Budget: ~$25–50. One afternoon on a 4x A100 node. Kill the instance when done —
`nvidia-smi` idle for 10 minutes means you're paying for nothing.

## 0. Vendor spike (do this FIRST, it decides the 1B capstone venue)

Compare, in a table in the phase note: Lambda / RunPod / Vast / Paperspace on
$/GPU-hr for 4x A100, 8x A100-80GB, 8x H100; spot-vs-on-demand; egress fees; how fast
instances actually provision. Decision criteria: 1B capstone ~ 1.2e20 FLOPs; at 35% MFU
that's ~12-14h on 8x H100 or ~38h on 8x A100-80GB. Under ~$400 total -> cloud capstone;
otherwise local RTX 6000 (~3-4 weeks with grad_checkpoint + compile).

## 1. Provision

- 4x A100 40GB node, Ubuntu + CUDA image. SSH key from ~/.ssh/id_ed25519.pub.
- rsync the repo (NOT data/shards — regenerate or scp the ~1GB tinystories shards, it's
  faster than re-tokenizing): `rsync -avz --exclude runs --exclude .git . ubuntu@NODE:microlab/`
- `pip install torch --index-url https://download.pytorch.org/whl/cu121` plus
  `pip install -e .` (or mirror the conda env; scripts only need torch + tokenizers +
  datasets + tensorboard).

## 2. DDP scaling drill (the point of the trip)

For N in 1, 2, 4:
    torchrun --nproc_per_node=N scripts/pretrain_ddp.py configs/150m.py \
        --data-dir data/shards/tinystories
Cap the run: temporarily set max_steps=120 in the config. Record tokens/sec from the TB
log (per-rank tps x N x batch x block x accum). Scaling efficiency = tps(N) / (N x tps(1)).
Expect ~0.9+ on one node; write down WHY it isn't 1.0 (gradient all-reduce overlap).

## 3. FSDP taste (optional, same node)

torchrun --nproc_per_node=4 with configs/1b.py and ZeRO-3-style sharding via
torch.distributed.fsdp — predict per-GPU memory with your Phase 7 memory_budget()
FIRST, then check nvidia-smi against the prediction. The delta IS the lesson.

## 4. Teardown

Download runs/ TB logs (rsync back), terminate the instance, verify billing stopped.
