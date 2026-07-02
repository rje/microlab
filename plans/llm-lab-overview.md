# Microlab LLM Curriculum Overview

## Goal

Build a single-GPU "mini frontier lab" for learning how modern LLMs are made. The program covers the same broad phases used by large labs - data, pretraining, scaling, post-training, preference optimization, reinforcement learning, distillation, tool use, and evaluation - but scales each phase to an RTX 6000 48GB server.

When forced to choose, optimize for deep understanding over speed. Each phase should include at least one exercise where the mechanism is implemented or inspected directly before relying on a high-level training framework.

## Working Assumptions

- Project root: `~/src/python/microlab`
- Hardware target: one RTX 6000-class GPU with 48GB VRAM
- Primary stack: Python, PyTorch, Hugging Face datasets/transformers/accelerate, PEFT, TRL, bitsandbytes
- Model sizes:
  - From scratch: start around 10M-50M parameters, then scale toward 100M-300M and a ~1B capstone
  - Full fine-tuning: small models, roughly 0.5B-3B depending on context length and optimizer
  - LoRA/QLoRA: 7B-14B comfortably, 30B as a stretch, 70B-class only as a careful experiment
- Preferred style: small reproducible experiments, written notes, plots, evals, and postmortems
- Cloud budget: hundreds of dollars total is acceptable for multi-GPU educational runs (Phase 7 drills ~$25-50; 1B capstone ~$300-400 if the vendor spike favors cloud); thousands is not.

## Track Design

Run two tracks in parallel.

### Track A: From-Scratch Understanding

This track uses tiny models so every part of the system can be understood, modified, and measured.

1. Build a dataset pipeline.
2. Train a tokenizer.
3. Implement and train a tiny decoder-only transformer.
4. Run architecture ablations.
5. Run scaling experiments.
6. Produce a model card and eval report.

### Track B: Modern Lab Practice

This track uses open base models and current post-training methods to mirror practical lab workflows.

1. Continue pretraining a small open model on a domain corpus.
2. Supervised fine-tune with instruction data.
3. Scale fine-tuning with LoRA and QLoRA.
4. Build preference data.
5. Train with DPO/ORPO/KTO.
6. Train reward models and run RL on verifiable tasks.
7. Distill reasoning traces into smaller models.
8. Train tool-use behavior and evaluate structured outputs.

## Phase Plan

### Phase 0: Evaluation Harness First

Build a lightweight evaluation harness before training anything serious. Include held-out prompts, exact-match tasks, JSON validity checks, retrieval/ranking tasks if used, simple coding tasks, latency, VRAM, and cost/time logs.

Deliverables:

- `evals/` with reproducible tasks
- baseline results for at least one open model
- a run log format for comparing experiments

Key readings:

- MMLU
- BIG-bench
- HELM
- HumanEval / Codex
- Chatbot Arena

### Phase 1: Data and Tokenization

Build a small corpus pipeline. Start with a public-domain or technical corpus, then add deduplication, quality filters, train/validation/test splits, contamination checks, and tokenizer training.

Deliverables:

- raw, cleaned, and tokenized dataset snapshots
- tokenizer comparison notes
- contamination and deduplication report

Key readings:

- BPE for rare words
- SentencePiece
- The Pile
- Deduplicating Training Data Makes Language Models Better
- DataComp-LM

### Phase 2: Tiny GPT Pretraining

Train a decoder-only transformer from scratch. Start small enough that mistakes are cheap. Track loss curves, sample quality, checkpoint behavior, and overfitting.

Deliverables:

- minimal GPT implementation or carefully inspected nanoGPT-style implementation
- first trained checkpoint
- sampling notebook
- loss and sample-quality report

Key readings:

- Attention Is All You Need
- GPT-3
- Scaling Laws
- Chinchilla

### Phase 3: Architecture Ablations

Change one architectural feature at a time and compare behavior. Suggested ablations: learned positions vs RoPE, LayerNorm vs RMSNorm, GELU MLP vs SwiGLU, standard attention vs FlashAttention where available, context length changes.

Deliverables:

- ablation matrix
- plots of validation loss and throughput
- grouped-query attention (GQA/MQA) and a tiny top-k MoE with load-balance loss
- short notes explaining which changes mattered and why

Key readings:

- RoPE
- RMSNorm
- GLU Variants Improve Transformer
- FlashAttention
- Switch Transformers
- Fast Transformer Decoding: One Write-Head is All You Need (MQA, Shazeer 2019)
- GQA: Training Generalized Multi-Query Transformer Models (Ainslie 2023)

### Phase 4: Scaling Experiments

Train a small family of models across model size, token budget, and data mix. The goal is not a strong model; the goal is to see scaling laws and bottlenecks in miniature.

Deliverables:

- at least three model sizes
- token budget comparison
- compute/time/VRAM table
- muP hyperparameter transfer table + a coordinate check across widths
- summary of where the server bottlenecks

Key readings:

- Scaling Laws
- Chinchilla
- LLaMA
- Llama 3 Herd of Models
- muP: Tensor Programs V (Feature Learning in Infinite-Width Neural Networks)
- Small-scale proxies for large-scale Transformer training instabilities

### Phase 5: Interpretability

Open up the trained 150M checkpoint and find real structure instead of trusting the loss
curve. Decode every layer's residual stream with the model's own unembedding (logit lens),
read attention patterns, and score induction heads on repeated sequences. Stretch goal:
watch induction heads form across saved checkpoints to see in-context learning emerge.

Deliverables:

- logit-lens decoder over all layers
- induction-head score on repeated-sequence probes
- attention-pattern visualizations on the 150M checkpoint
- short interpretability report

Key readings:

- Tuned Lens
- ROME (Locating and Editing Factual Associations in GPT)
- In-context Learning and Induction Heads (transformer-circuits.pub)

### Phase 6: Inference Engineering

Build everything between a checkpoint and a served token. Implement a KV cache and prove
generation is memory-bound, measure the KV-cache shrink that motivates GQA, quantize to
int8/int4 and pay the perplexity cost, and add speculative decoding with a draft model to
make decoding faster for free.

Deliverables:

- KV-cached generate (graded by exact token-match against uncached generation)
- sampling zoo (temperature, top-k, top-p, typical)
- groupwise int8/int4 quantizer + perplexity table
- speculative-decoding accept-rate benchmark on the 150M checkpoint

Key readings:

- PagedAttention (vLLM)
- Fast Inference from Transformers via Speculative Decoding
- GPTQ

### Phase 7: Distributed Training

Learn the parallelism vocabulary of every frontier lab - DP/TP/PP, ZeRO, FSDP - and feel it
on rented multi-GPU hardware. Hand-write the per-GPU memory budget, prove it against
nvidia-smi on a rented 4x A100 node (~$25-50), measure DDP scaling efficiency on the real
150M training script, and open with a vendor-affordability spike that decides whether the
1B capstone trains in the cloud (~12-14h on 8x H100) or locally (~3-4 weeks).

Deliverables:

- per-GPU memory budget (DP/TP/PP x ZeRO) validated against nvidia-smi
- DDP scaling-efficiency measurement on the 150M training script
- gradient-checkpointing and torch.compile drills
- vendor-affordability spike + the 1B capstone run

Key readings:

- Megatron-LM
- ZeRO

### Phase 8: Continued Pretraining

Take an open base model and continue next-token training on a domain corpus. Compare it against the base model and watch for catastrophic forgetting.

Deliverables:

- domain corpus
- continued-pretraining run
- base-vs-adapted evals
- forgetting analysis
- long-context extension via RoPE position interpolation

Key readings:

- LLaMA
- OPT
- Llama 3 Herd of Models
- DeepSeek-V3
- Extending Context Window of Large Language Models via Position Interpolation (Chen 2023)

### Phase 9: Supervised Fine-Tuning

Create instruction/response data and train a chat model. Use this phase to understand chat templates, prompt formats, synthetic data quality, and response style control.

Deliverables:

- SFT dataset
- SFT adapter or checkpoint
- before/after evals
- examples of improvements and regressions

Key readings:

- FLAN
- Scaling Instruction-Finetuned Language Models
- Self-Instruct
- InstructGPT

### Phase 10: Efficient Fine-Tuning

Use LoRA and QLoRA to scale from small fine-tuning to 7B-30B-class models on one GPU. Compare rank, target modules, quantization, learning rate, and sequence length.

Deliverables:

- LoRA baseline
- QLoRA baseline
- adapter comparison table
- merged or deployable adapter artifacts

Key readings:

- LoRA
- QLoRA

### Phase 11: Preference Data and Reward Models

Generate multiple completions per prompt and build chosen/rejected pairs. Train a small reward model and inspect where it fails.

Deliverables:

- preference dataset
- reward model
- reward calibration plots
- examples of reward hacking or misranking

Key readings:

- Fine-Tuning Language Models from Human Preferences
- InstructGPT
- Constitutional AI

### Phase 12: Offline Preference Optimization

Train with DPO, ORPO, and/or KTO. Compare against SFT using the same prompts and eval suite. Focus on how preference optimization changes helpfulness, verbosity, refusal behavior, and style.

Deliverables:

- DPO run
- one alternate method, such as ORPO or KTO
- SFT-vs-preference eval report
- qualitative failure analysis

Key readings:

- DPO
- ORPO
- KTO

### Phase 13: RL on Verifiable Tasks

Use tasks with automatic rewards: math answers, code tests, JSON schema validity, tool-call correctness, or small games. Start with a small model and keep the reward simple.

Deliverables:

- verifiable task environment
- RL training run
- reward curve
- pass-rate eval
- reward-hacking postmortem

Key readings:

- PPO
- DeepSeekMath
- DeepSeek-R1
- Training Verifiers to Solve Math Word Problems
- Let's Verify Step by Step

### Phase 14: Reasoning and Distillation

Use a stronger teacher model or a larger local model to generate solutions, traces, or rejected candidates. Distill into a smaller model, then test whether the student actually improves.

Deliverables:

- teacher-generated data
- filtered distillation dataset
- distilled model or adapter
- reasoning eval report

Key readings:

- STaR
- Process and Outcome Feedback
- DeepSeek-R1
- s1: Simple Test-Time Scaling

### Phase 15: Tool Use and Agentic Behavior

Train or fine-tune a model to emit structured tool calls. Use exact schema validation and environment feedback. This is one of the easiest places to measure real behavioral improvement.

Deliverables:

- tool-call dataset
- schema-validity eval
- task-success eval
- trained adapter

Key readings:

- ReAct
- Toolformer
- Gorilla
- ToolLLM

### Phase 16: Final Report

Write a capstone report that ties the work together. Include what improved, what failed, what was compute-bound, and which methods were worth the complexity.

Deliverables:

- final model card
- final eval table
- recommended next experiments
- reading notes index

## Suggested Cadence

For each phase:

1. Read one core paper before starting.
2. Build the smallest possible experiment.
3. Run the experiment and record results.
4. Read one or two deeper papers after you have felt the problem directly.
5. Write a short phase note: assumptions, setup, results, mistakes, and next move.

## Repository Shape

The layout separates **code** (`src/microlab/`, importable and tested) from
**data and configs** (`evals/`, `configs/`) and **generated artifacts**
(`runs/`, which is git-ignored). Phase work adds new packages under
`src/microlab/` and new data/config directories alongside the existing ones.

```text
microlab/
  plans/                      # project plans and setup notes
    llm-lab-overview.md
    2026-06-18-phase-0-evaluation-harness-plan.md
  papers/                     # local paper library (source of truth: manifest.json)
    README.md
    manifest.json
    foundations/ tokenizers-data/ modern-llm-recipes/ architecture/
    instruction-tuning/ efficient-finetuning/ preferences-rlhf/
    reasoning-rl/ tool-use/ evaluation/
  src/microlab/               # importable package code
    evals/                    # Phase 0 evaluation harness
  evals/                      # eval data (not code)
    suites/                   # JSONL eval suites
  configs/                    # run configs
    eval/                     # backend/model configs for the harness
  tests/                      # pytest suites mirroring src/
  notes/                      # reading notes and phase write-ups
  site/                       # Microlab Console (React) + content/
  ops/                        # nginx/systemd deployment templates
  scripts/                    # utility and run scripts
  data/ tokenizers/ training/ # added in later phases
  runs/                       # generated run artifacts (git-ignored)
```

## First Three Concrete Moves

1. Build the paper library and skim the foundations/data papers.
2. Implement the evaluation harness with a baseline open model.
3. Build the data/tokenizer pipeline and train the first tiny GPT.

## Explicitly Out of Scope

Decided exclusions, not omissions: multimodality (would double the curriculum),
RAG/retrieval systems (application-layer, not model-building), and deep safety work
(red-teaming, jailbreak evaluation) beyond the Constitutional AI reading. Revisit after
Phase 16 if interest survives contact with the 1B capstone.
