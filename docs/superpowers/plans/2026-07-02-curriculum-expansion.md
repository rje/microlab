# Curriculum Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three new curriculum phases (Interpretability, Inference Engineering, Distributed Training), extend four existing phases (GQA/MoE ablations, muP, long context), and add 12 papers — per the approved spec `docs/superpowers/specs/2026-07-02-curriculum-expansion-design.md`.

**Architecture:** Every addition follows the repo's four-layer pattern: reference oracle in `src/microlab/<area>/reference/` (tested, green on main), hand-write stub in `src/microlab/exercises/` (exercise-marked tests, red until solved, deselected from guardrail), a START-HERE guide in `docs/hand-write/`, and run-for-real scripts where applicable. Phases 5–13 renumber to 8–16 to make room.

**Tech Stack:** PyTorch, pytest (`exercise` marker convention), Flask console (`site/content/phases.json` drives phase pages), `scripts/download_papers.py` for the paper library.

## Global Constraints

- Python env: run everything via `/home/rje/anaconda3/bin/conda run -n microlab <cmd>` from repo root `/home/rje/src/python/microlab`.
- Ruff line length 100 (`ruff check` must pass; pre-commit runs ruff+pytest+vitest).
- Exercise tests end with `pytestmark = pytest.mark.exercise` and are EXPECTED to fail with NotImplementedError until the user solves them; they are deselected from the default run. The default `pytest tests/ --ignore=tests/exercises` (and the pre-commit guardrail) must stay green after every commit.
- The live 150M training run (`microlab-train-150m` systemd service) must keep working: NO change may alter `VariantGPT` default-path outputs or checkpoint key names. Task 8 asserts bit-for-bit compat.
- Commit messages end with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_018aPzfDwRdjhAE3v6CaCryE`
- `tests/exercises/test_phase10_rl.py`, `test_phase11_reasoning.py`, `test_phase12_tools.py` currently FAIL when run with `-m exercise` (unsolved stubs) — that is normal, not breakage.

---

### Task 1: Renumber phases 5–13 → 8–16

**Files:**
- Rename: `src/microlab/exercises/phase{05..12}_*.py` → `phase{08..15}_*.py`
- Rename: `tests/exercises/test_phase{05..12}_*.py` → `test_phase{08..15}_*.py`
- Rename: `docs/hand-write/phase{5..12}-*.md` → `phase{8..15}-*.md`
- Modify: `site/content/phases.json` (ids + titles for old phases 5–13)
- Modify: `docs/curriculum.md` (phase table + exercise-numbering prose)

**Interfaces:**
- Produces: module names `microlab.exercises.phase08_continued`, `phase09_sft`, `phase10_lora`, `phase11_reward`, `phase12_dpo`, `phase13_rl`, `phase14_reasoning`, `phase15_tools`; phases.json ids `phase-8`…`phase-16`. Later tasks (5–12) rely on these exact names.

- [ ] **Step 1: Rename exercise modules, tests, and guides (descending order to avoid collisions)**

```bash
cd /home/rje/src/python/microlab
git mv src/microlab/exercises/phase12_tools.py     src/microlab/exercises/phase15_tools.py
git mv src/microlab/exercises/phase11_reasoning.py src/microlab/exercises/phase14_reasoning.py
git mv src/microlab/exercises/phase10_rl.py        src/microlab/exercises/phase13_rl.py
git mv src/microlab/exercises/phase09_dpo.py       src/microlab/exercises/phase12_dpo.py
git mv src/microlab/exercises/phase08_reward.py    src/microlab/exercises/phase11_reward.py
git mv src/microlab/exercises/phase07_lora.py      src/microlab/exercises/phase10_lora.py
git mv src/microlab/exercises/phase06_sft.py       src/microlab/exercises/phase09_sft.py
git mv src/microlab/exercises/phase05_continued.py src/microlab/exercises/phase08_continued.py
git mv tests/exercises/test_phase12_tools.py     tests/exercises/test_phase15_tools.py
git mv tests/exercises/test_phase11_reasoning.py tests/exercises/test_phase14_reasoning.py
git mv tests/exercises/test_phase10_rl.py        tests/exercises/test_phase13_rl.py
git mv tests/exercises/test_phase09_dpo.py       tests/exercises/test_phase12_dpo.py
git mv tests/exercises/test_phase08_reward.py    tests/exercises/test_phase11_reward.py
git mv tests/exercises/test_phase07_lora.py      tests/exercises/test_phase10_lora.py
git mv tests/exercises/test_phase06_sft.py       tests/exercises/test_phase09_sft.py
git mv tests/exercises/test_phase05_continued.py tests/exercises/test_phase08_continued.py
git mv docs/hand-write/phase12-tools.md     docs/hand-write/phase15-tools.md
git mv docs/hand-write/phase11-reasoning.md docs/hand-write/phase14-reasoning.md
git mv docs/hand-write/phase10-rl.md        docs/hand-write/phase13-rl.md
git mv docs/hand-write/phase9-dpo.md        docs/hand-write/phase12-dpo.md
git mv docs/hand-write/phase8-reward.md     docs/hand-write/phase11-reward.md
git mv docs/hand-write/phase7-lora.md       docs/hand-write/phase10-lora.md
git mv docs/hand-write/phase6-sft.md        docs/hand-write/phase9-sft.md
git mv docs/hand-write/phase5-continued.md  docs/hand-write/phase8-continued.md
```
(If any `docs/hand-write` filename differs, `ls docs/hand-write/` and match the same +3 pattern.)

- [ ] **Step 2: Fix every stale reference to the old module/guide/phase numbers**

Find them all, then edit each hit (imports in tests, docstrings saying "(Phase 5)" etc., `-k phaseNN` hints, guide cross-references):

```bash
grep -rn "phase05_continued\|phase06_sft\|phase07_lora\|phase08_reward\|phase09_dpo\|phase10_rl\|phase11_reasoning\|phase12_tools" src/ tests/ docs/ site/content/
grep -rn "phase5-continued\|phase6-sft\|phase7-lora\|phase8-reward\|phase9-dpo\|phase10-rl\|phase11-reasoning\|phase12-tools" src/ tests/ docs/
```

Mapping (old → new) for module names, `-k` selectors, guide paths, and "(Phase N)" docstring labels inside the renamed files:
continued 5→8, sft 6→9, lora 7→10, reward 8→11, dpo 9→12, rl 10→13, reasoning 11→14, tools 12→15. Update the docstring text "(Phase 8)" etc. in each renamed exercise + its reference-oracle docstring if it names the phase (e.g. `model/reference/continued.py` says "Phase 5" — change to "Phase 8"; same for `sft.py`, `lora.py`, `reward.py`, `dpo.py`, `rl.py`, `reasoning.py`, `tools.py`, and `evals/` files if grep hits).

- [ ] **Step 3: Renumber `site/content/phases.json`**

For each entry with id `phase-5` … `phase-13`: id becomes `phase-8` … `phase-16` and the title's number shifts +3 (e.g. `"id": "phase-5", "title": "Phase 5: Continued Pretraining"` → `"id": "phase-8", "title": "Phase 8: Continued Pretraining"`). Do NOT add new phase entries yet (that's Task 12). Entries stay in file order.

- [ ] **Step 4: Update `docs/curriculum.md` phase table**

Replace the table rows for 5–13 with the same rows renumbered 8–16, and update the exercise-file range prose (`numbered phase00…phase12` → `numbered phase00…phase15`). Leave rows 0–4 untouched; new-phase rows land in Task 12.

- [ ] **Step 5: Verify green**

```bash
conda run -n microlab python -m pytest tests/ --ignore=tests/exercises -q
conda run -n microlab python -m pytest tests/exercises -m exercise -q 2>&1 | tail -3
conda run -n microlab python -c "
import sys; sys.path.insert(0, 'src')
from microlab.console.app import create_app
app = create_app(); c = app.test_client()
r = c.get('/login'); assert r.status_code == 200, r.status_code
print('console loads OK')"
conda run -n microlab ruff check src tests
```
Expected: main suite passes; exercise run shows the SAME pass/fail counts as before the rename (rename must not change results); console loads; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "refactor: renumber phases 5-13 to 8-16 to make room for interp/inference/distributed"
```

---

### Task 2: Add 12 papers to the library

**Files:**
- Modify: `papers/manifest.json` (append 12 entries)
- Create (via script): PDFs under `papers/{architecture,foundations,interpretability,inference,systems}/`, regenerated `papers/README.md`

**Interfaces:**
- Produces: paper ids (slugified titles) used by Task 12's `readingPaperIds`:
  `fast-transformer-decoding-one-write-head-is-all-you-need`, `gqa-training-generalized-multi-query-transformer-models-from-multi-head-checkpoints`, `extending-context-window-of-large-language-models-via-positional-interpolation`, `tensor-programs-v-tuning-large-neural-networks-via-zero-shot-hyperparameter-transfer`, `small-scale-proxies-for-large-scale-transformer-training-instabilities`, `eliciting-latent-predictions-from-transformers-with-the-tuned-lens`, `locating-and-editing-factual-associations-in-gpt`, `efficient-memory-management-for-large-language-model-serving-with-pagedattention`, `fast-inference-from-transformers-via-speculative-decoding`, `gptq-accurate-post-training-quantization-for-generative-pre-trained-transformers`, `megatron-lm-training-multi-billion-parameter-language-models-using-model-parallelism`, `zero-memory-optimizations-toward-training-trillion-parameter-models`

- [ ] **Step 1: Append the 12 manifest entries** (before the closing `]`, matching existing schema exactly):

```json
{
  "topic": "architecture",
  "title": "Fast Transformer Decoding: One Write-Head is All You Need",
  "authors": "Shazeer",
  "year": 2019,
  "source_url": "https://arxiv.org/abs/1911.02150",
  "pdf_url": "https://arxiv.org/pdf/1911.02150",
  "filename": "2019-shazeer-fast-transformer-decoding-one-write-head-is-all-you-need.pdf"
},
{
  "topic": "architecture",
  "title": "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
  "authors": "Ainslie et al.",
  "year": 2023,
  "source_url": "https://arxiv.org/abs/2305.13245",
  "pdf_url": "https://arxiv.org/pdf/2305.13245",
  "filename": "2023-ainslie-gqa-training-generalized-multi-query-transformer-models.pdf"
},
{
  "topic": "architecture",
  "title": "Extending Context Window of Large Language Models via Positional Interpolation",
  "authors": "Chen et al.",
  "year": 2023,
  "source_url": "https://arxiv.org/abs/2306.15595",
  "pdf_url": "https://arxiv.org/pdf/2306.15595",
  "filename": "2023-chen-extending-context-window-via-positional-interpolation.pdf"
},
{
  "topic": "foundations",
  "title": "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer",
  "authors": "Yang et al.",
  "year": 2022,
  "source_url": "https://arxiv.org/abs/2203.03466",
  "pdf_url": "https://arxiv.org/pdf/2203.03466",
  "filename": "2022-yang-tensor-programs-v-zero-shot-hyperparameter-transfer.pdf"
},
{
  "topic": "foundations",
  "title": "Small-scale proxies for large-scale Transformer training instabilities",
  "authors": "Wortsman et al.",
  "year": 2023,
  "source_url": "https://arxiv.org/abs/2309.14322",
  "pdf_url": "https://arxiv.org/pdf/2309.14322",
  "filename": "2023-wortsman-small-scale-proxies-for-training-instabilities.pdf"
},
{
  "topic": "interpretability",
  "title": "Eliciting Latent Predictions from Transformers with the Tuned Lens",
  "authors": "Belrose et al.",
  "year": 2023,
  "source_url": "https://arxiv.org/abs/2303.08112",
  "pdf_url": "https://arxiv.org/pdf/2303.08112",
  "filename": "2023-belrose-eliciting-latent-predictions-with-the-tuned-lens.pdf"
},
{
  "topic": "interpretability",
  "title": "Locating and Editing Factual Associations in GPT",
  "authors": "Meng et al.",
  "year": 2022,
  "source_url": "https://arxiv.org/abs/2202.05262",
  "pdf_url": "https://arxiv.org/pdf/2202.05262",
  "filename": "2022-meng-locating-and-editing-factual-associations-in-gpt.pdf"
},
{
  "topic": "inference",
  "title": "Efficient Memory Management for Large Language Model Serving with PagedAttention",
  "authors": "Kwon et al.",
  "year": 2023,
  "source_url": "https://arxiv.org/abs/2309.06180",
  "pdf_url": "https://arxiv.org/pdf/2309.06180",
  "filename": "2023-kwon-efficient-memory-management-with-pagedattention.pdf"
},
{
  "topic": "inference",
  "title": "Fast Inference from Transformers via Speculative Decoding",
  "authors": "Leviathan et al.",
  "year": 2022,
  "source_url": "https://arxiv.org/abs/2211.17192",
  "pdf_url": "https://arxiv.org/pdf/2211.17192",
  "filename": "2022-leviathan-fast-inference-via-speculative-decoding.pdf"
},
{
  "topic": "inference",
  "title": "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers",
  "authors": "Frantar et al.",
  "year": 2022,
  "source_url": "https://arxiv.org/abs/2210.17323",
  "pdf_url": "https://arxiv.org/pdf/2210.17323",
  "filename": "2022-frantar-gptq-accurate-post-training-quantization.pdf"
},
{
  "topic": "systems",
  "title": "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism",
  "authors": "Shoeybi et al.",
  "year": 2019,
  "source_url": "https://arxiv.org/abs/1909.08053",
  "pdf_url": "https://arxiv.org/pdf/1909.08053",
  "filename": "2019-shoeybi-megatron-lm-model-parallelism.pdf"
},
{
  "topic": "systems",
  "title": "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models",
  "authors": "Rajbhandari et al.",
  "year": 2019,
  "source_url": "https://arxiv.org/abs/1910.02054",
  "pdf_url": "https://arxiv.org/pdf/1910.02054",
  "filename": "2019-rajbhandari-zero-memory-optimizations.pdf"
}
```

- [ ] **Step 2: Download + verify**

```bash
conda run -n microlab python -c "import json; json.load(open('papers/manifest.json')); print('valid json')"
conda run -n microlab python scripts/download_papers.py
```
Expected final line: `downloaded=12 skipped=50 failures=0 total=62`. If any arXiv download fails, retry once; a persistent failure is an error to surface, not to skip.

- [ ] **Step 3: Verify slugs match the Interfaces list**

```bash
conda run -n microlab python -c "
import sys; sys.path.insert(0, 'src'); import json
from microlab.console.content import paper_id_for
m = json.load(open('papers/manifest.json'))
for e in m[-12:]: print(paper_id_for(e))"
```
Expected: exactly the 12 ids listed in Interfaces above. If any differs, use the ACTUAL printed id in Task 12.

- [ ] **Step 4: Commit** (PDFs are tracked — the library is committed in this repo; confirm with `git status papers/` that PDFs aren't gitignored, then):

```bash
git add papers/ && git commit -m "feat: add 12 papers — GQA/MQA, muP, stability, interp, inference, Megatron/ZeRO"
```

---

### Task 3: Reading-workspace content for the 12 new papers

**Files:**
- Create: `content/papers/<paper-id>/overview.json` and `content/papers/<paper-id>/cards.json` for each of the 12 ids from Task 2
- Create: `site/content/synopses/curriculum-expansion.json` (12 entries)

**Interfaces:**
- Consumes: paper ids from Task 2.
- Produces: console reading content; no code interfaces.

**Note for the executor:** this is editorial content — dispatch ONE subagent per paper (they can read the PDF under `papers/<topic>/<filename>`). Each subagent must mirror the exact JSON schema of the existing examples: `content/papers/attention-is-all-you-need/overview.json` (keys: `paperId`, `generatedAt` = today, `depthSuggestion` ∈ skim|read|implement, `tldr`, `overview`, plus whatever other keys that file has — copy its full key set), `content/papers/attention-is-all-you-need/cards.json` (5–8 cards, ids `<paper-id>#N`), and one synopsis entry per paper keyed by paper id mirroring `site/content/synopses/phase-0.json` entry shape (`paperId`, `oneSentence`, `summary`, `coreIdeas`, plus the full key set of an existing entry).

- [ ] **Step 1: Generate the 36 files/entries via subagents** (one paper each; batch launches).
- [ ] **Step 2: Validate**

```bash
conda run -n microlab python - <<'EOF'
import json, sys
sys.path.insert(0, 'src')
from microlab.console.content import paper_id_for
ids = [paper_id_for(e) for e in json.load(open('papers/manifest.json'))[-12:]]
syn = json.load(open('site/content/synopses/curriculum-expansion.json'))
for i in ids:
    json.load(open(f'content/papers/{i}/overview.json'))
    json.load(open(f'content/papers/{i}/cards.json'))
    assert i in syn, f'missing synopsis: {i}'
print('all 12 papers have overview+cards+synopsis')
EOF
conda run -n microlab python -c "
import sys; sys.path.insert(0, 'src')
from microlab.console.app import create_app
create_app().test_client().get('/login')
print('console still loads')"
```

- [ ] **Step 3: Commit**

```bash
git add content/papers site/content/synopses && git commit -m "feat: reading content (overview/cards/synopses) for the 12 expansion papers"
```

---

### Task 4: Phase 3 — GQA oracle + exercise

**Files:**
- Modify: `src/microlab/model/reference/variants.py` (add `n_kv_head` to `VariantConfig`, add `GQAAttention`, wire into `VariantBlock`)
- Test: `tests/model/test_gqa.py` (oracle tests, NOT exercise-marked)
- Modify: `src/microlab/exercises/phase03_variants.py` (add `GQAAttention` stub)
- Modify: `tests/exercises/test_phase03_variants.py` (add exercise tests)
- Modify: `docs/hand-write/phase3-ablations.md` (add GQA section)

**Interfaces:**
- Produces: `VariantConfig.n_kv_head: int | None = None`; `microlab.model.reference.variants.GQAAttention(config)` with params `q_proj: Linear(n_embd, n_embd)`, `kv_proj: Linear(n_embd, 2*n_kv_head*head_dim)`, `c_proj: Linear(n_embd, n_embd)` (all `bias=config.bias`), rope buffers `rope_cos`/`rope_sin`, `forward(x: (B,T,C)) -> (B,T,C)`. `n_kv_head=None` → existing attention classes, unchanged. Task 8 later adds an optional `kv_cache` param to attention forwards.

- [ ] **Step 1: Write failing oracle tests** — `tests/model/test_gqa.py`:

```python
"""GQA reference oracle tests: shapes, MHA-equivalence, and checkpoint compatibility."""

import pytest
import torch

from microlab.model.reference.variants import (
    GQAAttention,
    RoPECausalSelfAttention,
    VariantConfig,
    VariantGPT,
)


def _cfg(n_kv_head=None):
    return VariantConfig(
        vocab_size=64, block_size=32, n_layer=2, n_head=6, n_embd=48,
        norm="rms", pos="rope", mlp="swiglu", n_kv_head=n_kv_head,
    )


@pytest.mark.parametrize("n_kv", [1, 2, 3, 6])
def test_gqa_output_shape(n_kv):
    attn = GQAAttention(_cfg(n_kv_head=n_kv)).eval()
    x = torch.randn(2, 16, 48)
    assert attn(x).shape == (2, 16, 48)


def test_gqa_with_all_heads_equals_mha():
    # n_kv_head == n_head must reproduce the fused-projection MHA exactly when the
    # fused c_attn weights are sliced into q_proj / kv_proj.
    torch.manual_seed(0)
    cfg = _cfg(n_kv_head=6)
    mha = RoPECausalSelfAttention(cfg).eval()
    gqa = GQAAttention(cfg).eval()
    C = cfg.n_embd
    with torch.no_grad():
        gqa.q_proj.weight.copy_(mha.c_attn.weight[:C])
        gqa.kv_proj.weight.copy_(mha.c_attn.weight[C:])
        gqa.c_proj.weight.copy_(mha.c_proj.weight)
    x = torch.randn(2, 16, C)
    assert torch.allclose(gqa(x), mha(x), atol=1e-5)


def test_gqa_param_savings():
    full = sum(p.numel() for p in GQAAttention(_cfg(n_kv_head=6)).parameters())
    mqa = sum(p.numel() for p in GQAAttention(_cfg(n_kv_head=1)).parameters())
    assert mqa < full


def test_gqa_causality():
    attn = GQAAttention(_cfg(n_kv_head=2)).eval()
    x = torch.randn(1, 16, 48)
    y1 = attn(x)
    x2 = x.clone()
    x2[0, 10:] = 0.0  # changing the future...
    y2 = attn(x2)
    assert torch.allclose(y1[0, :10], y2[0, :10], atol=1e-6)  # ...can't change the past


def test_default_none_keeps_variantgpt_identical():
    # Checkpoint/live-run safety: n_kv_head=None must be bit-for-bit the old model.
    torch.manual_seed(0)
    a = VariantGPT(_cfg(n_kv_head=None))
    torch.manual_seed(0)
    b = VariantGPT(VariantConfig(vocab_size=64, block_size=32, n_layer=2, n_head=6,
                                 n_embd=48, norm="rms", pos="rope", mlp="swiglu"))
    x = torch.randint(0, 64, (2, 16))
    la, _ = a(x)
    lb, _ = b(x)
    assert torch.equal(la, lb)
    assert list(a.state_dict().keys()) == list(b.state_dict().keys())


def test_variantgpt_trains_with_gqa():
    m = VariantGPT(_cfg(n_kv_head=2))
    x = torch.randint(0, 64, (2, 16))
    _, loss = m(x, x)
    loss.backward()
    assert loss.isfinite()
```

- [ ] **Step 2: Run to verify failure**

`conda run -n microlab python -m pytest tests/model/test_gqa.py -q` → FAIL: `ImportError: cannot import name 'GQAAttention'`.

- [ ] **Step 3: Implement the oracle** in `src/microlab/model/reference/variants.py`:

Add to `VariantConfig`:
```python
@dataclass
class VariantConfig(GPTConfig):
    norm: str = "layer"   # "layer" | "rms"
    pos: str = "learned"  # "learned" | "rope"
    mlp: str = "gelu"     # "gelu" | "swiglu"
    # None -> classic multi-head attention (fused c_attn), bit-identical to before this
    # field existed. Set to a divisor of n_head for grouped-query attention (1 == MQA).
    n_kv_head: int | None = None
```

Add class (after `RoPECausalSelfAttention`):
```python
class GQAAttention(nn.Module):
    """Grouped-query attention with RoPE: n_head query heads share n_kv_head K/V heads
    (n_kv_head == 1 is multi-query attention). Halves-to-quarters the KV projection —
    and, later, the KV cache — at near-zero quality cost (Ainslie et al., 2023)."""

    def __init__(self, config: VariantConfig) -> None:
        super().__init__()
        assert config.pos == "rope", "GQAAttention is built for the RoPE block"
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.kv_proj = nn.Linear(
            config.n_embd, 2 * config.n_kv_head * self.head_dim, bias=config.bias
        )
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        cos, sin = build_rope_cache(config.block_size, self.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_head, self.head_dim)
        k = kv[:, :, 0].transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)
        q = apply_rope(q, self.rope_cos.to(q.dtype), self.rope_sin.to(q.dtype))
        k = apply_rope(k, self.rope_cos.to(k.dtype), self.rope_sin.to(k.dtype))
        groups = self.n_head // self.n_kv_head
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
```

Wire into `VariantBlock.__init__` (replace the existing `self.attn = ...` assignment):
```python
        if getattr(config, "n_kv_head", None) is not None:
            self.attn = GQAAttention(config)
        elif config.pos == "rope":
            self.attn = RoPECausalSelfAttention(config)
        else:
            self.attn = CausalSelfAttention(config)
```
NOTE the MHA-equivalence test: `kv_proj.weight` rows are ordered all-K-then-all-V (matching `c_attn.weight[C:]`), so `.view(B, T, 2, n_kv_head, head_dim)` is only correct when the projection output is laid out `[K..., V...]`. With the slice-copy that ordering means the view above must split as `kv[:, :, 0]`=K only if the reshape respects it — verify with the equivalence test; if it fails, use `k, v = self.kv_proj(x).split(self.n_kv_head * self.head_dim, dim=2)` then view each separately (this is the layout-safe form and is what the reference should ship).

- [ ] **Step 4: Run oracle tests** — `conda run -n microlab python -m pytest tests/model/test_gqa.py -q` → all PASS. Also run `conda run -n microlab python -m pytest tests/ --ignore=tests/exercises -q` → green (default-path unchanged).

- [ ] **Step 5: Add the exercise stub** to `src/microlab/exercises/phase03_variants.py` (append; extend the module docstring's list of primitives to four):

```python
class GQAAttention(nn.Module):
    """Grouped-query attention with RoPE. Same parameter names/shapes as the reference
    (``q_proj``, ``kv_proj``, ``c_proj``) so weights transfer via load_state_dict."""

    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_head % config.n_kv_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.kv_proj = nn.Linear(
            config.n_embd, 2 * config.n_kv_head * self.head_dim, bias=config.bias
        )
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        from microlab.model.reference.variants import build_rope_cache

        cos, sin = build_rope_cache(config.block_size, self.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "project q (n_head heads) and kv (n_kv_head heads), split k/v with "
            ".split(n_kv_head*head_dim, dim=2), apply RoPE to q and k, repeat_interleave "
            "k/v by n_head//n_kv_head groups, causal SDPA, merge heads, c_proj"
        )
```

- [ ] **Step 6: Add exercise tests** to `tests/exercises/test_phase03_variants.py` (before the `pytestmark` line):

```python
from microlab.exercises.phase03_variants import GQAAttention
from microlab.model.reference.variants import GQAAttention as RefGQA
from microlab.model.reference.variants import VariantConfig


@pytest.mark.parametrize("n_kv", [1, 3, 6])
def test_gqa_matches_reference(n_kv):
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=64, block_size=32, n_layer=2, n_head=6, n_embd=48,
                        norm="rms", pos="rope", mlp="swiglu", n_kv_head=n_kv)
    ref, stu = RefGQA(cfg).eval(), GQAAttention(cfg).eval()
    stu.load_state_dict(ref.state_dict())
    x = torch.randn(2, 16, 48)
    assert torch.allclose(stu(x), ref(x), atol=1e-5)
```

- [ ] **Step 7: Verify** — `conda run -n microlab python -m pytest tests/exercises/test_phase03_variants.py -m exercise -q`: old tests keep their status; the new GQA tests FAIL with NotImplementedError (expected). Default suite still green.

- [ ] **Step 8: Update the guide** — in `docs/hand-write/phase3-ablations.md`, change "Three pieces" to "Four pieces" and append:

```markdown
4. **`GQAAttention.forward`** — grouped-query attention: `n_head` query heads share
   `n_kv_head` K/V heads. Project q normally; project kv at `2*n_kv_head*head_dim` and
   `.split(n_kv_head*head_dim, dim=2)`; RoPE on q,k; `repeat_interleave` k,v by
   `n_head//n_kv_head`; causal SDPA. `n_kv_head=1` is MQA (Shazeer 2019), `=n_head` is
   plain MHA. **Why it exists won't fully land until Phase 6**: the KV *cache* shrinks by
   `n_head/n_kv_head`, and at inference time the cache — not compute — is the bottleneck.
   Ablate it now (add `n_kv_head` to your ablation matrix: loss barely moves); measure the
   cache payoff when you build inference.
```

- [ ] **Step 9: Commit**

```bash
git add -A && git commit -m "feat(phase3): GQA — reference oracle, hand-write exercise, ablation guide"
```

---

### Task 5: Phase 3 — MoE oracle + exercise

**Files:**
- Create: `src/microlab/model/reference/moe.py`
- Test: `tests/model/test_moe.py` (oracle)
- Modify: `src/microlab/exercises/phase03_variants.py` (add `route_topk`, `load_balance_loss` stubs)
- Modify: `tests/exercises/test_phase03_variants.py` (exercise tests)
- Modify: `docs/hand-write/phase3-ablations.md`

**Interfaces:**
- Produces: `microlab.model.reference.moe`: `route_topk(router_logits: (N,E), k) -> (weights: (N,k) renormalized to sum 1, indices: (N,k))`; `load_balance_loss(router_probs: (N,E), expert_indices: (N,k)) -> scalar Tensor` (Switch aux: `E * sum_e f_e * P_e`); `MoEMLP(config, n_experts=4, k=2)` with `router: Linear(n_embd, n_experts, bias=False)`, `experts: ModuleList[SwiGLUMLP]`, `forward(x: (B,T,C)) -> (y: (B,T,C), aux_loss: scalar)`.

- [ ] **Step 1: Failing oracle tests** — `tests/model/test_moe.py`:

```python
"""MoE reference oracle tests: routing math, load-balance loss, end-to-end MoE MLP."""

import torch

from microlab.model.reference.moe import MoEMLP, load_balance_loss, route_topk
from microlab.model.reference.variants import VariantConfig


def test_route_topk_selects_best_and_renormalizes():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0], [-1.0, 0.0, 3.0, 1.0]])
    weights, idx = route_topk(logits, k=2)
    assert idx.tolist() == [[0, 1], [2, 3]]
    assert torch.allclose(weights.sum(-1), torch.ones(2), atol=1e-6)
    probs = torch.softmax(logits, dim=-1)
    expected0 = probs[0, [0, 1]] / probs[0, [0, 1]].sum()
    assert torch.allclose(weights[0], expected0, atol=1e-6)


def test_load_balance_loss_uniform_is_one():
    # Perfectly uniform routing gives the Switch aux loss its minimum value of 1.0.
    N, E = 64, 4
    probs = torch.full((N, E), 1.0 / E)
    idx = torch.arange(N).remainder(E).unsqueeze(1)  # round-robin dispatch, k=1
    loss = load_balance_loss(probs, idx)
    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-6)


def test_load_balance_loss_collapse_is_e():
    # Total collapse onto one expert scores E (the worst case), penalizing imbalance.
    N, E = 64, 4
    probs = torch.zeros(N, E)
    probs[:, 0] = 1.0
    idx = torch.zeros(N, 1, dtype=torch.long)
    assert torch.allclose(load_balance_loss(probs, idx), torch.tensor(float(E)), atol=1e-6)


def test_moe_mlp_forward_and_backward():
    cfg = VariantConfig(vocab_size=64, block_size=32, n_layer=1, n_head=4, n_embd=32)
    moe = MoEMLP(cfg, n_experts=4, k=2)
    x = torch.randn(2, 8, 32)
    y, aux = moe(x)
    assert y.shape == x.shape and aux.ndim == 0
    (y.mean() + 0.01 * aux).backward()
    assert moe.router.weight.grad is not None
```

- [ ] **Step 2: Verify failure** — `pytest tests/model/test_moe.py -q` → ImportError.

- [ ] **Step 3: Implement** `src/microlab/model/reference/moe.py`:

```python
"""Reference mixture-of-experts primitives (Phase 3): top-k routing, the Switch
load-balancing auxiliary loss, and a token-choice MoE MLP built from SwiGLU experts.
The oracle the owner diffs hand-written routing against."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from microlab.model.reference.variants import SwiGLUMLP


def route_topk(router_logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax over ALL experts, take the top-k probs per token, renormalize the kept
    probs to sum to 1. Returns (weights (N,k), expert indices (N,k))."""
    probs = F.softmax(router_logits, dim=-1)
    weights, indices = torch.topk(probs, k, dim=-1)
    return weights / weights.sum(-1, keepdim=True), indices


def load_balance_loss(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
    """Switch-Transformer auxiliary loss: E * sum_e f_e * P_e, where f_e is the fraction
    of dispatched (token, slot) assignments that went to expert e and P_e is the mean
    router probability for e. Equals 1.0 under perfectly uniform routing, E on collapse."""
    n_experts = router_probs.size(-1)
    one_hot = F.one_hot(expert_indices, n_experts).float()  # (N, k, E)
    f = one_hot.sum(dim=(0, 1)) / expert_indices.numel()
    p = router_probs.mean(0)
    return n_experts * torch.sum(f * p)


class MoEMLP(nn.Module):
    """Token-choice MoE feed-forward: each token is routed to its top-k SwiGLU experts,
    outputs combined with the renormalized router weights. Returns (y, aux_loss) — add
    `aux_coef * aux_loss` (typical coef ~0.01) to the training loss to keep experts busy."""

    def __init__(self, config, n_experts: int = 4, k: int = 2) -> None:
        super().__init__()
        self.n_experts, self.k = n_experts, k
        self.router = nn.Linear(config.n_embd, n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLUMLP(config) for _ in range(n_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        flat = x.view(-1, C)
        logits = self.router(flat)
        weights, indices = route_topk(logits, self.k)
        aux = load_balance_loss(F.softmax(logits, dim=-1), indices)
        y = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            for slot in range(self.k):
                mask = indices[:, slot] == e
                if mask.any():
                    y[mask] += weights[mask, slot, None] * expert(flat[mask])
        return y.view(B, T, C), aux
```

- [ ] **Step 4: Oracle tests pass** — `pytest tests/model/test_moe.py -q` → 4 passed; full non-exercise suite green.

- [ ] **Step 5: Exercise stubs** (append to `src/microlab/exercises/phase03_variants.py`):

```python
def route_topk(router_logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax over all experts, top-k per token, renormalize kept probs to sum to 1.
    Returns (weights (N,k), indices (N,k)). Graded vs microlab.model.reference.moe."""
    raise NotImplementedError("softmax -> topk -> renormalize")


def load_balance_loss(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
    """Switch aux loss: E * sum_e (fraction dispatched to e) * (mean router prob of e).
    1.0 when routing is uniform; E when it collapses onto one expert."""
    raise NotImplementedError("one-hot the indices; f_e over all (token, slot) pairs")
```

- [ ] **Step 6: Exercise tests** (append to `tests/exercises/test_phase03_variants.py` before `pytestmark`):

```python
from microlab.exercises.phase03_variants import load_balance_loss, route_topk
from microlab.model.reference import moe as ref_moe


def test_route_topk_matches_reference():
    torch.manual_seed(0)
    logits = torch.randn(32, 8)
    w_s, i_s = route_topk(logits, k=2)
    w_r, i_r = ref_moe.route_topk(logits, k=2)
    assert torch.equal(i_s, i_r) and torch.allclose(w_s, w_r, atol=1e-6)


def test_load_balance_loss_matches_reference():
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(64, 4), dim=-1)
    idx = torch.randint(0, 4, (64, 2))
    assert torch.allclose(
        load_balance_loss(probs, idx), ref_moe.load_balance_loss(probs, idx), atol=1e-6
    )
```

- [ ] **Step 7: Verify** — exercise run shows the new tests failing with NotImplementedError; default suite green; ruff clean.

- [ ] **Step 8: Guide** — append to `docs/hand-write/phase3-ablations.md`:

```markdown
## MoE (the second new primitive pair)

`route_topk` + `load_balance_loss` in the same exercise file, graded vs
`microlab.model.reference.moe`. The reference `MoEMLP` shows them assembled: every modern
frontier model (Mixtral, DeepSeek-V3) is an MoE; the entire trick is (a) route each token
to k of E experts, (b) penalize the router for collapsing onto favorites. The aux loss is
worth deriving by hand: why is uniform routing exactly 1.0? Why does collapse score E?
Stretch: swap `MoEMLP` in as a `VariantBlock` MLP and ablate it on TinyShakespeare.
```

- [ ] **Step 9: Commit**

```bash
git add -A && git commit -m "feat(phase3): MoE — top-k routing + Switch load-balance oracle and exercise"
```

---

### Task 6: Phase 4 — muP oracle + exercise + coordinate check

**Files:**
- Modify: `src/microlab/model/reference/scaling.py` (add `mup_multipliers`, `mup_attn_scale`)
- Test: `tests/model/test_mup.py` (oracle)
- Modify: `src/microlab/exercises/phase04_scaling.py` (stubs)
- Modify: `tests/exercises/test_phase04_scaling.py` (exercise tests — file exists; check actual name with `ls tests/exercises/ | grep phase04` and use it)
- Create: `scripts/mup_coord_check.py`
- Modify: `docs/hand-write/phase4-scaling.md`

**Interfaces:**
- Produces: `mup_multipliers(base_width: int, width: int) -> dict[str, float]` with EXACTLY these keys: `width_mult` (m = width/base), `hidden_lr_mult` (1/m), `hidden_init_std_mult` (1/sqrt(m)), `output_logit_mult` (1/m), `embedding_lr_mult` (1.0); `mup_attn_scale(head_dim: int) -> float` returning `1.0/head_dim` (muP's 1/d in place of SP's 1/sqrt(d)).

- [ ] **Step 1: Failing oracle tests** — `tests/model/test_mup.py`:

```python
"""muP oracle tests: the zero-shot hyperparameter-transfer scaling table."""

import math

from microlab.model.reference.scaling import mup_attn_scale, mup_multipliers


def test_identity_at_base_width():
    m = mup_multipliers(256, 256)
    assert all(math.isclose(v, 1.0) for v in m.values())


def test_doubling_width():
    m = mup_multipliers(256, 512)
    assert math.isclose(m["width_mult"], 2.0)
    assert math.isclose(m["hidden_lr_mult"], 0.5)
    assert math.isclose(m["hidden_init_std_mult"], 1 / math.sqrt(2))
    assert math.isclose(m["output_logit_mult"], 0.5)
    assert math.isclose(m["embedding_lr_mult"], 1.0)


def test_attn_scale_is_one_over_d():
    assert math.isclose(mup_attn_scale(64), 1 / 64)
```

- [ ] **Step 2: Verify failure** (ImportError), **Step 3: Implement** in `scaling.py`:

```python
def mup_multipliers(base_width: int, width: int) -> dict[str, float]:
    """muP (Tensor Programs V) transfer table, relative to a tuned base width: as width
    grows by m, hidden (matrix-like) Adam LRs shrink by 1/m, hidden init std by 1/sqrt(m),
    the output-logit multiplier by 1/m; embedding (vector-like) LR stays put. Tune once at
    base_width, transfer everywhere."""
    m = width / base_width
    return {
        "width_mult": m,
        "hidden_lr_mult": 1.0 / m,
        "hidden_init_std_mult": m**-0.5,
        "output_logit_mult": 1.0 / m,
        "embedding_lr_mult": 1.0,
    }


def mup_attn_scale(head_dim: int) -> float:
    """muP uses 1/d attention scaling instead of the standard 1/sqrt(d)."""
    return 1.0 / head_dim
```

- [ ] **Step 4: Oracle green.** **Step 5: Stubs** (append to `src/microlab/exercises/phase04_scaling.py`):

```python
def mup_multipliers(base_width: int, width: int) -> dict[str, float]:
    """The muP transfer table (keys: width_mult, hidden_lr_mult, hidden_init_std_mult,
    output_logit_mult, embedding_lr_mult). Graded vs microlab.model.reference.scaling."""
    raise NotImplementedError("m = width/base; hidden LR ~ 1/m; hidden init ~ 1/sqrt(m)")


def mup_attn_scale(head_dim: int) -> float:
    """muP attention temperature: 1/d, not 1/sqrt(d)."""
    raise NotImplementedError()
```

- [ ] **Step 6: Exercise tests** (append to the phase04 exercise test file, before `pytestmark`; add imports mirroring its existing style):

```python
def test_mup_multipliers_matches_reference():
    from microlab.exercises.phase04_scaling import mup_attn_scale, mup_multipliers
    from microlab.model.reference import scaling as ref
    for base, w in [(128, 128), (128, 512), (256, 1024)]:
        assert mup_multipliers(base, w) == pytest.approx(ref.mup_multipliers(base, w))
    assert mup_attn_scale(48) == pytest.approx(ref.mup_attn_scale(48))
```

- [ ] **Step 7: Coordinate-check script** — `scripts/mup_coord_check.py`:

```python
"""muP coordinate check: activation RMS across widths should stay flat under muP scaling
and drift under standard scaling. Run on CPU or GPU; prints a table, saves nothing.

    python scripts/mup_coord_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.model.reference.scaling import mup_multipliers  # noqa: E402
from microlab.model.reference.variants import VariantConfig, VariantGPT  # noqa: E402

BASE = 64
WIDTHS = [64, 128, 256, 512]
STEPS = 20
LR = 0.01


def final_block_rms(width: int, mup: bool) -> float:
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=256, block_size=64, n_layer=2, n_head=max(1, width // 32),
                        n_embd=width, norm="rms", pos="rope", mlp="swiglu")
    model = VariantGPT(cfg)
    mults = mup_multipliers(BASE, width)
    if mup:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.ndim == 2 and "wte" not in name:
                    p.mul_(mults["hidden_init_std_mult"])
    lr_hidden = LR * (mults["hidden_lr_mult"] if mup else 1.0)
    hidden = [p for n, p in model.named_parameters() if p.ndim == 2 and "wte" not in n]
    vector = [p for n, p in model.named_parameters() if not (p.ndim == 2 and "wte" not in n)]
    opt = torch.optim.Adam([
        {"params": hidden, "lr": lr_hidden},
        {"params": vector, "lr": LR},
    ])
    gen = torch.Generator().manual_seed(1)
    for _ in range(STEPS):
        x = torch.randint(0, 256, (8, 64), generator=gen)
        _, loss = model(x, x)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        x = torch.randint(0, 256, (8, 64), generator=gen)
        h = model.transformer.drop(model.transformer.wte(x))
        for block in model.transformer.h:
            h = block(h)
        return h.pow(2).mean().sqrt().item()


def main() -> None:
    print(f"{'width':>6} {'SP rms':>10} {'muP rms':>10}")
    for w in WIDTHS:
        print(f"{w:>6} {final_block_rms(w, mup=False):>10.3f} {final_block_rms(w, mup=True):>10.3f}")
    print("muP column should stay roughly flat; SP column should drift with width.")


if __name__ == "__main__":
    main()
```

Run it: `conda run -n microlab python scripts/mup_coord_check.py` — verify it executes and the muP column is flatter than SP (record the table in the commit message if it is; if it is NOT flatter, that's a finding to debug, not to skip).

- [ ] **Step 8: Guide** — append to `docs/hand-write/phase4-scaling.md`:

```markdown
## muP (new)

Hand-write `mup_multipliers` + `mup_attn_scale` — the zero-shot HP-transfer table from
Tensor Programs V. This is how you'll pick the 1B run's learning rate in Phase 7: tune at
a small width once, transfer by the table, don't re-sweep at $300/run. Then run
`python scripts/mup_coord_check.py` and stare at the two columns: activation RMS flat
across widths = the signature that transfer will work. Reading: muP paper + "Small-scale
proxies for large-scale training instabilities" (what loss spikes look like before you
meet one at 1B).
```

- [ ] **Step 9: Verify + commit** — default suite green, new exercise tests red-by-NotImplemented, ruff clean.

```bash
git add -A && git commit -m "feat(phase4): muP transfer table oracle + exercise + coordinate-check script"
```

---

### Task 7: Phase 5 — Interpretability package

**Files:**
- Create: `src/microlab/interp/__init__.py` (empty), `src/microlab/interp/reference/__init__.py` (empty), `src/microlab/interp/reference/lens.py`
- Test: `tests/interp/test_lens.py` (oracle)
- Create: `src/microlab/exercises/phase05_interp.py`, `tests/exercises/test_phase05_interp.py`
- Create: `scripts/interp_report.py`
- Create: `docs/hand-write/phase5-interp.md`

**Interfaces:**
- Produces: `microlab.interp.reference.lens`:
  - `collect_residual_stream(model: VariantGPT, idx: (B,T)) -> list[Tensor(B,T,C)]` — length `n_layer+1`: post-embedding(+dropout) state, then after each block; no grad.
  - `logit_lens(residuals: list[Tensor(B,T,C)], ln_f: nn.Module, lm_head: nn.Module) -> Tensor(L+1,B,T,V)`.
  - `attention_patterns(model: VariantGPT, idx: (1,T)) -> Tensor(n_layer, n_head, T, T)` — softmax attention probs recomputed from module weights (RoPE path).
  - `repeated_token_sequence(vocab_size: int, period: int, repeats: int, generator: torch.Generator) -> Tensor(1, period*repeats)`.
  - `induction_score(attn: Tensor(..., T, T), period: int) -> Tensor(...)` — mean over query positions `i >= period` of `attn[..., i, i - period + 1]`.

- [ ] **Step 1: Failing oracle tests** — `tests/interp/test_lens.py` (create `tests/interp/__init__.py` empty if other test dirs have one; check `ls tests/model/` for the convention and mirror it):

```python
"""Interp oracle tests: residual-stream collection, logit lens, induction scoring."""

import torch

from microlab.interp.reference.lens import (
    attention_patterns,
    collect_residual_stream,
    induction_score,
    logit_lens,
    repeated_token_sequence,
)
from microlab.model.reference.variants import VariantConfig, VariantGPT


def _model():
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=64, block_size=64, n_layer=3, n_head=4, n_embd=32,
                        norm="rms", pos="rope", mlp="swiglu")
    return VariantGPT(cfg).eval()


def test_residual_stream_shapes_and_final_equals_forward():
    m = _model()
    x = torch.randint(0, 64, (2, 10))
    res = collect_residual_stream(m, x)
    assert len(res) == 4 and all(r.shape == (2, 10, 32) for r in res)
    logits, _ = m(x)
    lens = logit_lens(res, m.transformer.ln_f, m.lm_head)
    assert lens.shape == (4, 2, 10, 64)
    assert torch.allclose(lens[-1], logits, atol=1e-5)  # last layer IS the model output


def test_attention_patterns_rows_sum_to_one_and_causal():
    m = _model()
    x = torch.randint(0, 64, (1, 12))
    attn = attention_patterns(m, x)
    assert attn.shape == (3, 4, 12, 12)
    assert torch.allclose(attn.sum(-1), torch.ones(3, 4, 12), atol=1e-5)
    assert torch.all(torch.triu(attn, diagonal=1).abs() < 1e-6)  # no attention to future


def test_induction_score_perfect_head_is_one():
    T, P = 12, 4
    attn = torch.zeros(1, T, T)
    for i in range(P, T):
        attn[0, i, i - P + 1] = 1.0  # a textbook induction head
    assert torch.allclose(induction_score(attn, P), torch.ones(1), atol=1e-6)


def test_repeated_sequence_repeats():
    g = torch.Generator().manual_seed(0)
    seq = repeated_token_sequence(64, period=8, repeats=3, generator=g)
    assert seq.shape == (1, 24)
    assert torch.equal(seq[0, :8], seq[0, 8:16]) and torch.equal(seq[0, :8], seq[0, 16:])
```

- [ ] **Step 2: Verify failure** (ModuleNotFoundError), **Step 3: Implement** `src/microlab/interp/reference/lens.py`:

```python
"""Reference interpretability tools (Phase 5): residual-stream capture, the logit lens,
attention-pattern extraction, and induction-head scoring — run against the from-scratch
models whose every weight we own. The oracle the owner diffs hand-written lenses against."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from microlab.model.reference.variants import VariantGPT, apply_rope


@torch.no_grad()
def collect_residual_stream(model: VariantGPT, idx: torch.Tensor) -> list[torch.Tensor]:
    """Mirror VariantGPT.forward, keeping the residual stream after embedding and after
    each block. Returns n_layer+1 tensors of shape (B, T, C)."""
    x = model.transformer.wte(idx)
    if model.config.pos == "learned":
        pos = torch.arange(idx.size(1), device=idx.device)
        x = x + model.transformer.wpe(pos)
    x = model.transformer.drop(x)
    stream = [x]
    for block in model.transformer.h:
        x = block(x)
        stream.append(x)
    return stream


@torch.no_grad()
def logit_lens(residuals, ln_f, lm_head) -> torch.Tensor:
    """Decode EVERY layer's residual state through the model's own final norm + unembed:
    what would the model predict if it had to stop here? Returns (L+1, B, T, V)."""
    return torch.stack([lm_head(ln_f(r)) for r in residuals])


@torch.no_grad()
def attention_patterns(model: VariantGPT, idx: torch.Tensor) -> torch.Tensor:
    """Recompute softmax attention probabilities per layer/head for the RoPE block (SDPA
    never materializes them). Returns (n_layer, n_head, T, T)."""
    assert model.config.pos == "rope", "attention_patterns supports the RoPE block"
    x = model.transformer.drop(model.transformer.wte(idx))
    B, T = idx.shape
    out = []
    for block in model.transformer.h:
        h = block.ln_1(x)
        a = block.attn
        if hasattr(a, "q_proj"):  # GQAAttention
            q = a.q_proj(h).view(B, T, a.n_head, a.head_dim).transpose(1, 2)
            k, _ = a.kv_proj(h).split(a.n_kv_head * a.head_dim, dim=2)
            k = k.view(B, T, a.n_kv_head, a.head_dim).transpose(1, 2)
            k = k.repeat_interleave(a.n_head // a.n_kv_head, dim=1)
        else:  # RoPECausalSelfAttention
            q, k, _ = a.c_attn(h).split(a.n_embd, dim=2)
            q = q.view(B, T, a.n_head, a.n_embd // a.n_head).transpose(1, 2)
            k = k.view(B, T, a.n_head, a.n_embd // a.n_head).transpose(1, 2)
        q = apply_rope(q, a.rope_cos.to(q.dtype), a.rope_sin.to(q.dtype))
        k = apply_rope(k, a.rope_cos.to(k.dtype), a.rope_sin.to(k.dtype))
        scores = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        out.append(F.softmax(scores, dim=-1)[0])
        x = block(x)
    return torch.stack(out)


def repeated_token_sequence(
    vocab_size: int, period: int, repeats: int, generator: torch.Generator
) -> torch.Tensor:
    """A random block of `period` tokens tiled `repeats` times — the classic induction
    probe: after the first repetition, [A B] ... [A ?] is predictable by copying."""
    block = torch.randint(0, vocab_size, (period,), generator=generator)
    return block.repeat(repeats).unsqueeze(0)


def induction_score(attn: torch.Tensor, period: int) -> torch.Tensor:
    """Mean attention mass on the induction target: from position i, the token AFTER the
    previous occurrence of the current token, i.e. offset i - period + 1. Scores near 1
    mean 'this head is an induction head'. attn: (..., T, T) -> (...)."""
    T = attn.size(-1)
    idx = torch.arange(period, T)
    return attn[..., idx, idx - period + 1].mean(-1)
```

- [ ] **Step 4: Oracle green** (`pytest tests/interp -q`), full suite green.

- [ ] **Step 5: Exercise stubs** — `src/microlab/exercises/phase05_interp.py`:

```python
"""Hand-write exercise (Phase 5): the two core interpretability primitives — the logit
lens and the induction-head score.

Fill in the ``NotImplementedError`` bodies so ``tests/exercises/test_phase05_interp.py``
passes. Graded against ``microlab.interp.reference.lens``. See
docs/hand-write/phase5-interp.md.
"""

from __future__ import annotations

import torch


def logit_lens(residuals: list[torch.Tensor], ln_f, lm_head) -> torch.Tensor:
    """Decode every layer's residual state through the model's own final norm + unembed.
    residuals: n_layer+1 tensors (B,T,C). Returns (L+1, B, T, V). The final layer's slice
    must equal the model's real output logits."""
    raise NotImplementedError("stack([lm_head(ln_f(r)) for r in residuals])")


def induction_score(attn: torch.Tensor, period: int) -> torch.Tensor:
    """Mean attention mass at offset (i - period + 1) over query positions i >= period.
    attn: (..., T, T) -> (...). A perfect induction head scores 1.0."""
    raise NotImplementedError("gather attn[..., i, i-period+1] for i in [period, T)")
```

- [ ] **Step 6: Exercise tests** — `tests/exercises/test_phase05_interp.py`:

```python
"""Spec + validation for the hand-written Phase-5 interpretability primitives."""

import pytest
import torch

from microlab.exercises.phase05_interp import induction_score, logit_lens
from microlab.interp.reference import lens as ref
from microlab.model.reference.variants import VariantConfig, VariantGPT


def _model():
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=64, block_size=64, n_layer=3, n_head=4, n_embd=32,
                        norm="rms", pos="rope", mlp="swiglu")
    return VariantGPT(cfg).eval()


def test_logit_lens_matches_reference_on_real_model():
    m = _model()
    x = torch.randint(0, 64, (2, 10))
    res = ref.collect_residual_stream(m, x)
    assert torch.allclose(
        logit_lens(res, m.transformer.ln_f, m.lm_head),
        ref.logit_lens(res, m.transformer.ln_f, m.lm_head), atol=1e-6,
    )


def test_induction_score_matches_reference():
    m = _model()
    g = torch.Generator().manual_seed(0)
    seq = ref.repeated_token_sequence(64, period=8, repeats=3, generator=g)
    attn = ref.attention_patterns(m, seq)
    assert torch.allclose(induction_score(attn, 8), ref.induction_score(attn, 8), atol=1e-6)

pytestmark = pytest.mark.exercise
```

- [ ] **Step 7: Report script** — `scripts/interp_report.py`:

```python
"""Interp report against a trained checkpoint: logit-lens progression for a prompt,
per-head induction scores, and attention heatmaps for the top induction heads.

    python scripts/interp_report.py runs/150m --data-dir data/shards/tinystories \
        --prompt "Once upon a time" --out runs/interp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.interp.reference.lens import (  # noqa: E402
    attention_patterns,
    collect_residual_stream,
    induction_score,
    logit_lens,
    repeated_token_sequence,
)
from microlab.model.reference.variants import VariantGPT  # noqa: E402
from microlab.tokenizer.fast import FastTokenizer  # noqa: E402


def load_model(run_dir: Path) -> VariantGPT:
    ckpts = sorted(run_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"no ckpt_*.pt in {run_dir}")
    ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    from microlab.model.reference.variants import VariantConfig

    model = VariantGPT(VariantConfig(
        vocab_size=cfg.vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
        n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=0.0, norm=cfg.norm, pos=cfg.pos,
        mlp=cfg.mlp,
    ))
    model.load_state_dict(ckpt["model"])
    print(f"loaded {ckpts[-1]} (step {ckpt['step']})")
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--data-dir", default="data/shards/tinystories")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--out", type=Path, default=Path("runs/interp"))
    args = ap.parse_args()

    tok = FastTokenizer.load(str(Path(args.data_dir) / "tokenizer.json"))
    model = load_model(args.run_dir)
    args.out.mkdir(parents=True, exist_ok=True)

    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
    res = collect_residual_stream(model, ids)
    lens = logit_lens(res, model.transformer.ln_f, model.lm_head)
    print("\nlogit lens — top-1 next-token prediction per layer (last position):")
    for layer, row in enumerate(lens[:, 0, -1, :]):
        top = row.argmax().item()
        print(f"  layer {layer:>2}: {tok.decode([top])!r}  (p={row.softmax(-1).max():.3f})")

    g = torch.Generator().manual_seed(0)
    seq = repeated_token_sequence(model.config.vocab_size, period=32, repeats=2, generator=g)
    attn = attention_patterns(model, seq)
    scores = induction_score(attn, 32)  # (n_layer, n_head)
    flat = [(s.item(), layer, h) for layer, row in enumerate(scores) for h, s in enumerate(row)]
    flat.sort(reverse=True)
    print("\ntop induction heads (score, layer, head):")
    for s, layer, h in flat[:5]:
        print(f"  {s:.3f}  L{layer} H{h}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for rank, (s, layer, h) in enumerate(flat[:3]):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(attn[layer, h].numpy(), cmap="viridis")
            ax.set_title(f"L{layer} H{h} induction={s:.3f}")
            fig.savefig(args.out / f"induction_{rank}_L{layer}H{h}.png", dpi=120)
            plt.close(fig)
        print(f"\nheatmaps -> {args.out}/")
    except ImportError:
        print("\nmatplotlib not installed — skipped heatmaps (scores above still stand)")


if __name__ == "__main__":
    main()
```

Run it for real once a `ckpt_*.pt` exists in `runs/150m`:
`conda run -n microlab python scripts/interp_report.py runs/150m` — verify it prints a lens table and induction scores (values are data, any values; the script running end-to-end is the check).

- [ ] **Step 8: Guide** — `docs/hand-write/phase5-interp.md`:

```markdown
> **Exercise — on `main`, no branch switching.** Implement the stub in
> `src/microlab/exercises/phase05_interp.py`, then run
> `pytest tests/exercises/test_phase05_interp.py -m exercise` to grade it.

# START HERE — look inside the model you trained (Phase 5)

You own every weight of the 150M TinyStories model. This phase is about seeing what those
weights learned. Two hand-writes, both graded against `microlab.interp.reference.lens`:

1. **`logit_lens`** — decode EVERY layer's residual stream through the model's own final
   norm + unembedding. Early layers predict garbage; late layers converge on the real
   output. Where does the answer "appear"? (Tuned Lens is the paper-grade version.)
2. **`induction_score`** — feed a repeated random sequence [A B C … A B C …]; an induction
   head at position i attends to i−period+1 (the token AFTER the previous occurrence of
   the current token) — that's copying, the mechanism behind in-context learning. Score =
   mean attention mass on that diagonal.

## See it on the real model

```bash
python scripts/interp_report.py runs/150m
```

## Stretch (uses a decision made during training)

Checkpoint pruning is off, so `runs/150m/ckpt_200.pt, ckpt_400.pt, …` all exist. Run the
induction scoring across checkpoints and plot max-score vs step: induction heads form in a
phase change (Anthropic, "In-context Learning and Induction Heads" —
https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/).

## Readings

Tuned Lens; ROME (locating factual associations). Both in the console reading workspace.
```

- [ ] **Step 9: Verify + commit** — oracle tests green, exercise tests red-by-NotImplemented, default suite green, ruff clean.

```bash
git add -A && git commit -m "feat(phase5): interpretability — logit lens + induction heads (oracle, exercise, report script)"
```

---

### Task 8: Phase 6 — KV cache (model path + oracle + exercise)

**Files:**
- Modify: `src/microlab/model/reference/variants.py` (optional `kv_cache` path through `RoPECausalSelfAttention`, `GQAAttention`, `VariantBlock`, `VariantGPT`)
- Create: `src/microlab/infer/__init__.py`, `src/microlab/infer/reference/__init__.py`, `src/microlab/infer/reference/kv_cache.py`
- Test: `tests/infer/test_kv_cache.py` (oracle)
- Create: `src/microlab/exercises/phase06_inference.py` (KV stub; Task 9 appends more), `tests/exercises/test_phase06_inference.py`

**Interfaces:**
- Produces: `microlab.infer.reference.kv_cache`:
  - `KVCache(n_layer, batch_size, n_kv_head, capacity, head_dim, dtype=torch.float32, device="cpu")` with `.seq_len: int` (tokens cached so far), `.append(layer: int, k: (B,nk,t,hd), v) -> (k_all: (B,nk,seq_len+t,hd), v_all)` (layer `n_layer-1`'s append advances `.seq_len`).
  - `generate_cached(model, idx: (B,T), max_new_tokens, temperature=0.0, top_k=None, generator=None) -> (B, T+max_new_tokens)`.
- Attention forwards gain `kv_cache: tuple[KVCache, int] | None = None` (the cache and this layer's index) and `rope_offset: int = 0`; `VariantGPT.forward` gains `kv_cache: KVCache | None = None`. ALL defaults None/0 → behavior byte-identical to today.

- [ ] **Step 1: Failing oracle tests** — `tests/infer/test_kv_cache.py`:

```python
"""KV-cache oracle: cached generation must EXACTLY match uncached generation, and be
usable in prefill + one-token-step decoding."""

import pytest
import torch

from microlab.infer.reference.kv_cache import KVCache, generate_cached
from microlab.model.reference.sample import generate
from microlab.model.reference.variants import VariantConfig, VariantGPT


def _model(n_kv_head=None):
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=64, block_size=64, n_layer=3, n_head=4, n_embd=32,
                        norm="rms", pos="rope", mlp="swiglu", n_kv_head=n_kv_head)
    return VariantGPT(cfg).eval()


@pytest.mark.parametrize("n_kv_head", [None, 2])
def test_cached_forward_matches_uncached(n_kv_head):
    m = _model(n_kv_head)
    x = torch.randint(0, 64, (2, 10))
    full_logits, _ = m(x)
    cache = KVCache(3, 2, n_kv_head or 4, 64, 8)
    pre_logits, _ = m(x[:, :6], kv_cache=cache)          # prefill
    assert torch.allclose(pre_logits, full_logits[:, :6], atol=1e-5)
    for t in range(6, 10):                                # one-token steps
        step_logits, _ = m(x[:, t:t + 1], kv_cache=cache)
        assert torch.allclose(step_logits[:, 0], full_logits[:, t], atol=1e-4)


def test_generate_cached_exactly_matches_reference_greedy():
    m = _model()
    idx = torch.randint(0, 64, (2, 8))
    assert torch.equal(
        generate_cached(m, idx.clone(), 20, temperature=0.0),
        generate(m, idx.clone(), 20, temperature=0.0),
    )


def test_default_path_untouched():
    m = _model()
    x = torch.randint(0, 64, (2, 10))
    torch.manual_seed(1)
    a, _ = m(x)
    torch.manual_seed(1)
    b, _ = m(x, kv_cache=None)
    assert torch.equal(a, b)
```

- [ ] **Step 2: Verify failure.** **Step 3: Implement.**

`src/microlab/infer/reference/kv_cache.py`:

```python
"""Reference KV cache (Phase 6): the single most important inference optimization.
Without it, generating token T recomputes attention keys/values for all T-1 previous
tokens; with it, each new token costs one forward over ONE position. This is also why
inference is memory-bound — the cache is (n_layer, B, n_kv_head, T, head_dim) big — and
why GQA (Phase 3) exists: fewer KV heads, smaller cache."""

from __future__ import annotations

import torch
from torch.nn import functional as F


class KVCache:
    """Preallocated per-layer K/V buffers. append() writes new keys/values at the current
    position and returns full views; the LAST layer's append advances seq_len (all layers
    see the same positions each step)."""

    def __init__(self, n_layer: int, batch_size: int, n_kv_head: int, capacity: int,
                 head_dim: int, dtype=torch.float32, device="cpu") -> None:
        self.n_layer = n_layer
        self.capacity = capacity
        self.seq_len = 0
        shape = (batch_size, n_kv_head, capacity, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layer)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layer)]

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        t = k.size(2)
        assert self.seq_len + t <= self.capacity, "KV cache overflow"
        self.k[layer][:, :, self.seq_len:self.seq_len + t] = k
        self.v[layer][:, :, self.seq_len:self.seq_len + t] = v
        k_all = self.k[layer][:, :, : self.seq_len + t]
        v_all = self.v[layer][:, :, : self.seq_len + t]
        if layer == self.n_layer - 1:
            self.seq_len += t
        return k_all, v_all


@torch.no_grad()
def generate_cached(model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.0,
                    top_k: int | None = None, generator: torch.Generator | None = None):
    """Autoregressive generation with a KV cache: one full prefill, then one-token steps.
    Greedy (temperature=0) output is token-for-token identical to the uncached
    microlab.model.reference.sample.generate."""
    model.eval()
    cfg = model.config
    n_kv = cfg.n_kv_head if getattr(cfg, "n_kv_head", None) else cfg.n_head
    cache = KVCache(cfg.n_layer, idx.size(0), n_kv, cfg.block_size,
                    cfg.n_embd // cfg.n_head, device=idx.device)
    logits, _ = model(idx, kv_cache=cache)
    for _ in range(max_new_tokens):
        step = logits[:, -1, :]
        if temperature == 0.0:
            nxt = step.argmax(dim=-1, keepdim=True)
        else:
            step = step / temperature
            if top_k is not None:
                v, _ = torch.topk(step, min(top_k, step.size(-1)))
                step[step < v[:, [-1]]] = -float("inf")
            probs = F.softmax(step, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator)
        idx = torch.cat((idx, nxt), dim=1)
        if cache.seq_len >= cfg.block_size:
            break  # context full — matches uncached crop-free semantics up to block_size
        logits, _ = model(nxt, kv_cache=cache)
    return idx
```

`variants.py` changes — thread the cache through (all-default = today's behavior):

In `RoPECausalSelfAttention.forward`, replace the signature and body:
```python
    def forward(self, x: torch.Tensor, kv_cache=None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        offset = kv_cache[0].seq_len if kv_cache is not None else 0
        cos = self.rope_cos[offset:offset + T].to(q.dtype)
        sin = self.rope_sin[offset:offset + T].to(q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if kv_cache is not None:
            cache, layer = kv_cache
            k, v = cache.append(layer, k, v)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=(q.size(-2) == k.size(-2)),
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
```
(`is_causal` note: with a cache, one-token steps have q_len=1 < k_len → attend to all — correct; prefill has q_len==k_len → causal — correct. Multi-token continuation after prefill is not supported; `generate_cached` never does it.)

Same pattern in `GQAAttention.forward`: compute `offset`, slice `cos/sin` with it, and between RoPE and `repeat_interleave` insert:
```python
        if kv_cache is not None:
            cache, layer = kv_cache
            k, v = cache.append(layer, k, v)
```
with the same `is_causal=(q.size(-2) == k.size(-2))` in SDPA.

`VariantBlock.forward`:
```python
    def forward(self, x: torch.Tensor, kv_cache=None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), kv_cache=kv_cache)
        x = x + self.mlp(self.ln_2(x))
        return x
```
BUT `CausalSelfAttention` (learned-pos path, from `gpt.py`) does not accept `kv_cache` — guard it:
```python
    def forward(self, x: torch.Tensor, kv_cache=None) -> torch.Tensor:
        if kv_cache is not None:
            x = x + self.attn(self.ln_1(x), kv_cache=kv_cache)
        else:
            x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```
(caching asserts `pos == "rope"` at the VariantGPT level below, so the learned path never receives one.)

`VariantGPT.forward`:
```python
    def forward(self, idx, targets=None, kv_cache=None):
        _, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size"
        if kv_cache is not None:
            assert self.config.pos == "rope", "KV cache requires the RoPE block"
        x = self.transformer.wte(idx)
        if self.config.pos == "learned":
            pos = torch.arange(T, device=idx.device)
            x = x + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        for i, block in enumerate(self.transformer.h):
            x = block(x, kv_cache=(kv_cache, i) if kv_cache is not None else None)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
```

- [ ] **Step 4: Oracle green + REGRESSION SWEEP** — `pytest tests/infer tests/model -q` then the FULL non-exercise suite (`pytest tests/ --ignore=tests/exercises -q`) must be green, including `test_resume_equivalence_cpu` and `test_default_none_keeps_variantgpt_identical` — this is the live-run-safety gate for the forward-signature change.

- [ ] **Step 5: Exercise stub** — create `src/microlab/exercises/phase06_inference.py`:

```python
"""Hand-write exercise (Phase 6): inference engineering — the KV cache, the sampling zoo,
groupwise quantization, and the speculative-decoding accept rule.

Fill in the ``NotImplementedError`` bodies so ``tests/exercises/test_phase06_inference.py``
passes. Graded against ``microlab.infer.reference``. See docs/hand-write/phase6-inference.md.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F  # noqa: F401  (you'll want it)


@torch.no_grad()
def generate_cached(model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.0,
                    top_k: int | None = None, generator: torch.Generator | None = None):
    """KV-cached generation: build a microlab.infer.reference.kv_cache.KVCache sized to
    the model config, prefill once with the full prompt, then feed ONE token at a time.
    Greedy output must EXACTLY match the uncached reference generate — and be faster."""
    raise NotImplementedError(
        "cache = KVCache(n_layer, B, n_kv_head or n_head, block_size, head_dim); "
        "logits, _ = model(idx, kv_cache=cache); then loop: pick next from logits[:, -1], "
        "append, model(next_token, kv_cache=cache)"
    )
```

- [ ] **Step 6: Exercise test** — create `tests/exercises/test_phase06_inference.py`:

```python
"""Spec + validation for the hand-written Phase-6 inference primitives."""

import time

import pytest
import torch

from microlab.model.reference.sample import generate
from microlab.model.reference.variants import VariantConfig, VariantGPT


def _model(block_size=256):
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=64, block_size=block_size, n_layer=4, n_head=4,
                        n_embd=64, norm="rms", pos="rope", mlp="swiglu")
    return VariantGPT(cfg).eval()


def test_generate_cached_exact_match_and_faster():
    from microlab.exercises.phase06_inference import generate_cached
    m = _model()
    idx = torch.randint(0, 64, (1, 8))
    t0 = time.perf_counter()
    ref = generate(m, idx.clone(), 200, temperature=0.0)
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    stu = generate_cached(m, idx.clone(), 200, temperature=0.0)
    t_stu = time.perf_counter() - t0
    assert torch.equal(stu, ref), "cached generation must be token-for-token identical"
    assert t_stu < t_ref, f"cache should be faster: cached={t_stu:.3f}s uncached={t_ref:.3f}s"

pytestmark = pytest.mark.exercise
```

- [ ] **Step 7: Verify + commit** — exercise test red-by-NotImplemented; everything else green; ruff clean.

```bash
git add -A && git commit -m "feat(phase6): KV cache — cached forward path through VariantGPT, oracle, exercise"
```

---

### Task 9: Phase 6 — sampling, quantization, speculative decoding + bench

**Files:**
- Create: `src/microlab/infer/reference/sampling.py`, `src/microlab/infer/reference/quant.py`, `src/microlab/infer/reference/speculative.py`
- Test: `tests/infer/test_sampling.py`, `tests/infer/test_quant.py`, `tests/infer/test_speculative.py`
- Modify: `src/microlab/exercises/phase06_inference.py`, `tests/exercises/test_phase06_inference.py` (append stubs + tests)
- Create: `scripts/bench_inference.py`, `docs/hand-write/phase6-inference.md`

**Interfaces:**
- Produces:
  - `sample_next(logits: (B,V), temperature=1.0, top_k=None, top_p=None, generator=None) -> (B,1)`. Order: temperature scale → top-k filter → top-p filter → softmax → multinomial. `temperature=0` → argmax. Top-p keeps the smallest descending-sorted prefix with cumulative prob ≥ p (always ≥ 1 token).
  - `quantize_groupwise(w: (out,in), bits=4, group_size=64) -> Tensor` — symmetric absmax round-trip (dequantized result). `in` must be divisible by group_size; qmax = 2^(bits-1) − 1; scale = group absmax / qmax; out = clamp(round(w/scale), −qmax, qmax) * scale.
  - `quantize_model_(model, bits=4, group_size=64) -> model` — in-place round-trip on every `nn.Linear.weight` whose in-features divide group_size.
  - `speculative_accept(draft_tokens: (K,), draft_probs: (K,V), target_probs: (K,V), generator) -> tuple[int, Tensor | None]` — Leviathan rule: accept token i with prob `min(1, p_t/p_d)`; at the first rejection return (n_accepted, one token resampled from `normalize(max(0, p_t − p_d))`); if all accepted return (K, None).

- [ ] **Step 1: Failing oracle tests** — three files:

`tests/infer/test_sampling.py`:
```python
import torch

from microlab.infer.reference.sampling import sample_next


def test_temperature_zero_is_argmax():
    logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 0.0]])
    assert sample_next(logits, temperature=0.0).squeeze(1).tolist() == [1, 0]


def test_top_k_restricts_support():
    torch.manual_seed(0)
    logits = torch.tensor([[5.0, 4.0, -10.0, -10.0]])
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        tok = sample_next(logits, top_k=2, generator=g).item()
        assert tok in (0, 1)


def test_top_p_keeps_minimal_prefix():
    # probs ~ [0.7, 0.2, 0.06, 0.04]; top_p=0.8 keeps exactly {0, 1}
    probs = torch.tensor([[0.7, 0.2, 0.06, 0.04]])
    logits = probs.log()
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        assert sample_next(logits, top_p=0.8, generator=g).item() in (0, 1)


def test_generator_reproducible():
    logits = torch.randn(2, 16)
    a = sample_next(logits, generator=torch.Generator().manual_seed(7))
    b = sample_next(logits, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
```

`tests/infer/test_quant.py`:
```python
import torch

from microlab.infer.reference.quant import quantize_groupwise, quantize_model_
from microlab.model.reference.variants import VariantConfig, VariantGPT


def test_round_trip_error_small_and_shrinks_with_bits():
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    e8 = (quantize_groupwise(w, bits=8) - w).abs().mean()
    e4 = (quantize_groupwise(w, bits=4) - w).abs().mean()
    assert e8 < e4 < w.abs().mean()  # int8 beats int4 beats garbage


def test_group_scales_are_local():
    w = torch.ones(1, 128)
    w[0, :64] = 100.0  # a huge first group must not wreck the second group's precision
    q = quantize_groupwise(w, bits=4, group_size=64)
    assert (q[0, 64:] - 1.0).abs().max() < 0.2


def test_quantize_model_runs_and_changes_weights():
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=64, block_size=32, n_layer=1, n_head=4, n_embd=64,
                        norm="rms", pos="rope", mlp="swiglu")
    m = VariantGPT(cfg)
    before = m.transformer.h[0].attn.c_attn.weight.clone()
    quantize_model_(m, bits=4, group_size=32)
    after = m.transformer.h[0].attn.c_attn.weight
    assert not torch.equal(before, after)
    x = torch.randint(0, 64, (1, 8))
    logits, _ = m(x)
    assert logits.isfinite().all()
```

`tests/infer/test_speculative.py`:
```python
import torch

from microlab.infer.reference.speculative import speculative_accept


def test_identical_distributions_accept_everything():
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(4, 16), dim=-1)
    tokens = probs.argmax(-1)
    n, fix = speculative_accept(tokens, probs, probs, torch.Generator().manual_seed(0))
    assert n == 4 and fix is None


def test_target_zero_prob_rejects_at_that_position():
    V = 8
    draft = torch.full((2, V), 1.0 / V)
    target = draft.clone()
    tokens = torch.tensor([3, 5])
    target[0, 3] = 0.0  # target hates the first draft token
    target[0] /= target[0].sum()
    n, fix = speculative_accept(tokens, draft, target, torch.Generator().manual_seed(0))
    assert n == 0 and fix is not None and fix.item() != 3


def test_resample_comes_from_positive_residual():
    V = 4
    draft = torch.tensor([[0.97, 0.01, 0.01, 0.01]])
    target = torch.tensor([[0.01, 0.49, 0.25, 0.25]])
    tokens = torch.tensor([0])
    g = torch.Generator().manual_seed(0)
    for _ in range(30):
        n, fix = speculative_accept(tokens, draft, target, g)
        if n == 0:
            assert fix.item() != 0  # residual max(0, p_t - p_d) is zero at token 0
```

- [ ] **Step 2: Verify failures.** **Step 3: Implement the three modules:**

`src/microlab/infer/reference/sampling.py`:
```python
"""Reference next-token sampling (Phase 6): temperature, top-k, and top-p (nucleus) in
the standard order — scale, filter, filter, softmax, sample."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                top_p: float | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """logits (B, V) -> next token ids (B, 1). temperature=0 is greedy argmax."""
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # drop tokens once the cumulative prob BEFORE them already reached top_p
        drop = (cum - F.softmax(sorted_logits, dim=-1)) >= top_p
        sorted_logits = sorted_logits.masked_fill(drop, -float("inf"))
        logits = torch.full_like(logits, -float("inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)
```

`src/microlab/infer/reference/quant.py`:
```python
"""Reference groupwise quantization (Phase 6): symmetric absmax round-trip, the shape of
every weight-only inference quant scheme (GPTQ/AWQ add smarter rounding on top)."""

from __future__ import annotations

import torch


def quantize_groupwise(w: torch.Tensor, bits: int = 4, group_size: int = 64) -> torch.Tensor:
    """Quantize-dequantize each `group_size` slice of the input dim independently.
    Returns the dequantized tensor (same shape/dtype) so quality impact is measurable."""
    out_f, in_f = w.shape
    assert in_f % group_size == 0, f"in_features {in_f} not divisible by {group_size}"
    qmax = 2 ** (bits - 1) - 1
    groups = w.view(out_f, in_f // group_size, group_size)
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(groups / scale), -qmax, qmax)
    return (q * scale).view(out_f, in_f)


@torch.no_grad()
def quantize_model_(model: torch.nn.Module, bits: int = 4, group_size: int = 64):
    """In-place round-trip every Linear weight whose in_features divide group_size."""
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module.weight.size(1) % group_size == 0:
            module.weight.copy_(quantize_groupwise(module.weight, bits, group_size))
    return model
```

`src/microlab/infer/reference/speculative.py`:
```python
"""Reference speculative-decoding accept rule (Phase 6, Leviathan et al. 2022): a cheap
draft model proposes K tokens; the target model verifies them in ONE forward. Accepting
with prob min(1, p_target/p_draft) and resampling rejections from the positive residual
provably samples from the target distribution — free speedup, zero quality loss."""

from __future__ import annotations

import torch


def speculative_accept(draft_tokens: torch.Tensor, draft_probs: torch.Tensor,
                       target_probs: torch.Tensor, generator: torch.Generator):
    """draft_tokens (K,), draft/target probs (K, V). Returns (n_accepted, correction):
    correction is a token sampled from normalize(max(0, target - draft)) at the first
    rejected position, or None when all K drafts are accepted."""
    k = draft_tokens.size(0)
    for i in range(k):
        tok = draft_tokens[i]
        p_d = draft_probs[i, tok].clamp(min=1e-12)
        p_t = target_probs[i, tok]
        u = torch.rand((), generator=generator)
        if u <= torch.clamp(p_t / p_d, max=1.0):
            continue
        residual = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
        residual = residual / residual.sum().clamp(min=1e-12)
        fix = torch.multinomial(residual, 1, generator=generator)[0]
        return i, fix
    return k, None
```

- [ ] **Step 4: Oracle tests green; full suite green.**

- [ ] **Step 5: Append exercise stubs** to `src/microlab/exercises/phase06_inference.py`:

```python
def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                top_p: float | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """(B,V) -> (B,1). Order: temperature -> top-k filter -> top-p filter -> softmax ->
    multinomial(generator). temperature=0 -> argmax. Top-p: sort desc, keep the smallest
    prefix whose cumulative prob reaches p (never drop the top token)."""
    raise NotImplementedError()


def quantize_groupwise(w: torch.Tensor, bits: int = 4, group_size: int = 64) -> torch.Tensor:
    """Symmetric absmax quantize-dequantize per group along the input dim. qmax =
    2**(bits-1) - 1; scale = group_absmax/qmax; round, clamp to [-qmax, qmax], rescale."""
    raise NotImplementedError()


def speculative_accept(draft_tokens: torch.Tensor, draft_probs: torch.Tensor,
                       target_probs: torch.Tensor, generator: torch.Generator):
    """Leviathan accept/reject: accept draft i with prob min(1, p_t/p_d); at the first
    rejection return (i, token resampled from normalize(max(0, p_t - p_d))); if all K
    accepted return (K, None)."""
    raise NotImplementedError()
```

- [ ] **Step 6: Append exercise tests** to `tests/exercises/test_phase06_inference.py` (before `pytestmark`):

```python
def test_sample_next_matches_reference():
    from microlab.exercises.phase06_inference import sample_next
    from microlab.infer.reference.sampling import sample_next as ref_sample
    torch.manual_seed(0)
    logits = torch.randn(4, 32)
    for kwargs in [dict(temperature=0.0), dict(temperature=0.8, top_k=5),
                   dict(temperature=1.0, top_p=0.9), dict(temperature=0.7, top_k=8, top_p=0.95)]:
        a = sample_next(logits.clone(), generator=torch.Generator().manual_seed(3), **kwargs)
        b = ref_sample(logits.clone(), generator=torch.Generator().manual_seed(3), **kwargs)
        assert torch.equal(a, b), kwargs


def test_quantize_groupwise_matches_reference():
    from microlab.exercises.phase06_inference import quantize_groupwise
    from microlab.infer.reference.quant import quantize_groupwise as ref_q
    torch.manual_seed(0)
    w = torch.randn(32, 128)
    for bits in (4, 8):
        assert torch.allclose(quantize_groupwise(w, bits=bits), ref_q(w, bits=bits), atol=1e-6)


def test_speculative_accept_matches_reference():
    from microlab.exercises.phase06_inference import speculative_accept
    from microlab.infer.reference.speculative import speculative_accept as ref_acc
    torch.manual_seed(0)
    for seed in range(10):
        draft = torch.softmax(torch.randn(4, 16), -1)
        target = torch.softmax(torch.randn(4, 16), -1)
        tokens = torch.multinomial(draft, 1).squeeze(1)
        a = speculative_accept(tokens, draft, target, torch.Generator().manual_seed(seed))
        b = ref_acc(tokens, draft, target, torch.Generator().manual_seed(seed))
        assert a[0] == b[0]
        assert (a[1] is None) == (b[1] is None)
        if a[1] is not None:
            assert torch.equal(a[1], b[1])
```

- [ ] **Step 7: Bench script** — `scripts/bench_inference.py`:

```python
"""Inference bench against a trained checkpoint: tok/s uncached vs KV-cached vs
cached+int8, perplexity before/after quantization, and the GQA cache-size table.

    python scripts/bench_inference.py runs/150m --data-dir data/shards/tinystories
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.data.shard_dataset import ShardDataset  # noqa: E402
from microlab.evals.perplexity import evaluate_perplexity  # noqa: E402
from microlab.infer.reference.kv_cache import generate_cached  # noqa: E402
from microlab.infer.reference.quant import quantize_model_  # noqa: E402
from microlab.model.reference.sample import generate  # noqa: E402

from scripts.interp_report import load_model  # noqa: E402


def bench(fn, *args, n=3, **kwargs) -> float:
    fn(*args, **kwargs)  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(*args, **kwargs)
    dt = (time.perf_counter() - t0) / n
    return (out.size(1) * out.size(0)) / dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--data-dir", default="data/shards/tinystories")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--new-tokens", type=int, default=256)
    args = ap.parse_args()

    model = load_model(args.run_dir).to(args.device)
    idx = torch.zeros((1, 8), dtype=torch.long, device=args.device)

    tps_slow = bench(generate, model, idx, args.new_tokens, temperature=0.0)
    tps_fast = bench(generate_cached, model, idx, args.new_tokens, temperature=0.0)
    print(f"uncached: {tps_slow:8.1f} tok/s")
    print(f"KV cache: {tps_fast:8.1f} tok/s  ({tps_fast / tps_slow:.1f}x)")

    val = ShardDataset(args.data_dir, split="val")
    ppl = evaluate_perplexity(model, val, model.config.block_size, 8, iters=50,
                              device=args.device)
    q8 = quantize_model_(copy.deepcopy(model), bits=8)
    q4 = quantize_model_(copy.deepcopy(model), bits=4)
    ppl8 = evaluate_perplexity(q8, val, model.config.block_size, 8, iters=50,
                               device=args.device)
    ppl4 = evaluate_perplexity(q4, val, model.config.block_size, 8, iters=50,
                               device=args.device)
    print(f"perplexity: fp32={ppl:.2f}  int8={ppl8:.2f}  int4={ppl4:.2f}")

    cfg = model.config
    hd = cfg.n_embd // cfg.n_head
    for n_kv in (cfg.n_head, cfg.n_head // 2, 1):
        by = 2 * cfg.n_layer * n_kv * cfg.block_size * hd * 2  # k+v, bf16 bytes
        print(f"KV cache @ n_kv_head={n_kv:>2}: {by / 1e6:7.1f} MB per sequence")


if __name__ == "__main__":
    main()
```

Note: `from scripts.interp_report import load_model` only works if `scripts/` is importable — it isn't a package. Instead duplicate the small `load_model` helper into this script (copy the function body from Task 7's `interp_report.py`) and drop that import. Run for real once a checkpoint exists:
`conda run -n microlab python scripts/bench_inference.py runs/150m` — expect a >1× cache speedup, int8 ppl ≈ fp32 ppl, int4 ppl slightly worse, and a 4× cache-size drop at n_kv_head=3 in the table.

- [ ] **Step 8: Guide** — `docs/hand-write/phase6-inference.md`:

```markdown
> **Exercise — on `main`, no branch switching.** Implement the stubs in
> `src/microlab/exercises/phase06_inference.py`, then run
> `pytest tests/exercises/test_phase06_inference.py -m exercise` to grade them.

# START HERE — inference engineering (Phase 6)

Everything between a checkpoint and a served token. Four hand-writes, graded against
`microlab.infer.reference`:

1. **`generate_cached`** — the KV cache. Without it, token T recomputes K/V for all T−1
   predecessors (generation is O(T²)); with it, each step is one single-token forward.
   Graded by EXACT token-match against the uncached reference + a measured speedup. This
   is the sharpest test in the curriculum: off-by-one RoPE offsets produce subtly-wrong
   text, and exact-match catches what "looks right" misses.
2. **`sample_next`** — temperature, top-k, top-p in the standard order. Fixed-seed graded.
3. **`quantize_groupwise`** — symmetric absmax per group; the skeleton under GPTQ/AWQ.
4. **`speculative_accept`** — the accept/reject rule that makes a draft model free: accept
   with min(1, p_t/p_d), resample rejections from max(0, p_t − p_d). Phase 14 closes the
   loop: your distilled student IS a draft model.

## Run it for real

```bash
python scripts/bench_inference.py runs/150m
```

tok/s uncached vs cached vs quantized; perplexity cost of int8/int4; and the payoff of
Phase 3's GQA: the KV-cache-bytes table (n_kv_head 12 -> 3 = 4x smaller cache).

## Readings

PagedAttention (what vLLM does when many sequences share a GPU), Speculative Decoding,
GPTQ. All in the console.
```

- [ ] **Step 9: Verify + commit.**

```bash
git add -A && git commit -m "feat(phase6): sampling, groupwise quant, speculative accept — oracles, exercises, bench"
```

---

### Task 10: Phase 7 — Distributed training

**Files:**
- Create: `src/microlab/distributed/__init__.py`, `src/microlab/distributed/reference/__init__.py`, `src/microlab/distributed/reference/memory.py`
- Test: `tests/distributed/test_memory.py` (oracle)
- Create: `src/microlab/exercises/phase07_distributed.py`, `tests/exercises/test_phase07_distributed.py`
- Modify: `src/microlab/train/config.py` (add `grad_checkpoint`, `compile` flags), `src/microlab/train/trainer.py` (honor them; checkpoint via raw model), `src/microlab/model/reference/variants.py` (checkpointed block loop)
- Test: additions to `tests/train/test_trainer.py`
- Create: `scripts/pretrain_ddp.py`, `ops/lambda-distributed.md`, `docs/hand-write/phase7-distributed.md`

**Interfaces:**
- Produces: `memory_budget(n_params: int, n_layer: int, n_embd: int, block_size: int, micro_batch: int, dp: int = 1, tp: int = 1, pp: int = 1, zero_stage: int = 0, grad_checkpoint: bool = False, dtype_bytes: int = 2) -> dict[str, float]` with keys `params`, `grads`, `optimizer`, `activations`, `total` (bytes per GPU). Pinned formulas:
  - `params = n_params*dtype_bytes/(tp*pp)`, additionally `/dp` iff `zero_stage >= 3`
  - `grads = n_params*dtype_bytes/(tp*pp)`, additionally `/dp` iff `zero_stage >= 2`
  - `optimizer = n_params*12/(tp*pp)` (fp32 master + Adam m + v), additionally `/dp` iff `zero_stage >= 1`
  - `activations = (n_layer/pp) * micro_batch * block_size * n_embd * dtype_bytes * (1 if grad_checkpoint else 34) / tp`  — the 34 is the standard no-recompute per-layer multiplier (documented approximation); with checkpointing only layer inputs are stored.
  - `total` = sum.
- `RunConfig.grad_checkpoint: bool = False`, `RunConfig.compile: bool = False`.

- [ ] **Step 1: Failing oracle tests** — `tests/distributed/test_memory.py`:

```python
"""Memory-budget oracle: the closed-form bookkeeping behind 'will it fit'."""

import pytest

from microlab.distributed.reference.memory import memory_budget

GB = 1e9
ONE_B = dict(n_params=1_000_000_000, n_layer=24, n_embd=1792, block_size=1024,
             micro_batch=16)


def test_single_gpu_baseline():
    b = memory_budget(**ONE_B)
    assert b["params"] == pytest.approx(2 * ONE_B["n_params"])
    assert b["optimizer"] == pytest.approx(12 * ONE_B["n_params"])
    assert b["total"] == sum(b[k] for k in ("params", "grads", "optimizer", "activations"))


def test_zero_stages_shed_state_monotonically():
    budgets = [memory_budget(**ONE_B, dp=8, zero_stage=z)["total"] for z in (0, 1, 2, 3)]
    assert budgets[0] > budgets[1] > budgets[2] > budgets[3]
    z3 = memory_budget(**ONE_B, dp=8, zero_stage=3)
    assert z3["params"] == pytest.approx(2 * ONE_B["n_params"] / 8)


def test_tp_divides_everything():
    a = memory_budget(**ONE_B)
    b = memory_budget(**ONE_B, tp=2)
    for key in ("params", "grads", "optimizer", "activations"):
        assert b[key] == pytest.approx(a[key] / 2)


def test_grad_checkpoint_slashes_activations():
    a = memory_budget(**ONE_B)["activations"]
    b = memory_budget(**ONE_B, grad_checkpoint=True)["activations"]
    assert b == pytest.approx(a / 34)


def test_dp_alone_shrinks_nothing_at_zero0():
    assert memory_budget(**ONE_B)["total"] == pytest.approx(
        memory_budget(**ONE_B, dp=8)["total"])
```

- [ ] **Step 2: Verify failure.** **Step 3: Implement** `src/microlab/distributed/reference/memory.py`:

```python
"""Reference per-GPU memory budget (Phase 7): where the bytes go when training with
data/tensor/pipeline parallelism and ZeRO. Closed-form and approximate on activations
(the 34-bytes-per-element multiplier is the standard no-recompute transformer estimate);
the cloud drills verify it against nvidia-smi reality."""

from __future__ import annotations

ACT_MULT_NO_CKPT = 34  # ~= attention + MLP intermediates per layer, bf16, no recompute


def memory_budget(n_params: int, n_layer: int, n_embd: int, block_size: int,
                  micro_batch: int, dp: int = 1, tp: int = 1, pp: int = 1,
                  zero_stage: int = 0, grad_checkpoint: bool = False,
                  dtype_bytes: int = 2) -> dict[str, float]:
    """Bytes per GPU for {params, grads, optimizer, activations, total}. Model state is
    split by tp*pp; ZeRO additionally shards optimizer (>=1), grads (>=2), params (>=3)
    across dp. Optimizer assumes AdamW with fp32 master weights (12 bytes/param)."""
    shard = tp * pp
    params = n_params * dtype_bytes / shard
    grads = n_params * dtype_bytes / shard
    optimizer = n_params * 12 / shard
    if zero_stage >= 1:
        optimizer /= dp
    if zero_stage >= 2:
        grads /= dp
    if zero_stage >= 3:
        params /= dp
    act_mult = 1 if grad_checkpoint else ACT_MULT_NO_CKPT
    activations = (n_layer / pp) * micro_batch * block_size * n_embd * dtype_bytes * act_mult / tp
    total = params + grads + optimizer + activations
    return {"params": params, "grads": grads, "optimizer": optimizer,
            "activations": activations, "total": total}
```

- [ ] **Step 4: Oracle green.** **Step 5: Exercise stub** — `src/microlab/exercises/phase07_distributed.py`:

```python
"""Hand-write exercise (Phase 7): the per-GPU memory budget — the closed-form bookkeeping
every lab does before renting a cluster.

Fill in the ``NotImplementedError`` body so ``tests/exercises/test_phase07_distributed.py``
passes. Graded against ``microlab.distributed.reference.memory``. See
docs/hand-write/phase7-distributed.md.
"""

from __future__ import annotations


def memory_budget(n_params: int, n_layer: int, n_embd: int, block_size: int,
                  micro_batch: int, dp: int = 1, tp: int = 1, pp: int = 1,
                  zero_stage: int = 0, grad_checkpoint: bool = False,
                  dtype_bytes: int = 2) -> dict[str, float]:
    """Keys: params, grads, optimizer, activations, total (bytes per GPU).
    Model state / (tp*pp); ZeRO shards optimizer(>=1), grads(>=2), params(>=3) over dp.
    AdamW fp32 master = 12 bytes/param. Activations: (n_layer/pp) * micro_batch *
    block_size * n_embd * dtype_bytes * (1 if ckpt else 34) / tp."""
    raise NotImplementedError()
```

- [ ] **Step 6: Exercise test** — `tests/exercises/test_phase07_distributed.py`:

```python
"""Spec + validation for the hand-written Phase-7 memory budget."""

import itertools

import pytest

from microlab.distributed.reference.memory import memory_budget as ref_budget
from microlab.exercises.phase07_distributed import memory_budget


def test_memory_budget_matches_reference_across_matrix():
    base = dict(n_params=7_000_000_000, n_layer=32, n_embd=4096, block_size=2048,
                micro_batch=4)
    for dp, tp, pp, z, ck in itertools.product((1, 8), (1, 2), (1, 4), (0, 1, 2, 3),
                                               (False, True)):
        got = memory_budget(**base, dp=dp, tp=tp, pp=pp, zero_stage=z, grad_checkpoint=ck)
        want = ref_budget(**base, dp=dp, tp=tp, pp=pp, zero_stage=z, grad_checkpoint=ck)
        assert got == pytest.approx(want), (dp, tp, pp, z, ck)

pytestmark = pytest.mark.exercise
```

- [ ] **Step 7: RunConfig + Trainer + VariantGPT flags (TDD)** — add to `tests/train/test_trainer.py`:

```python
def test_grad_checkpoint_and_compile_flags(tmp_path):
    # Both flags must run a short training and still checkpoint/resume via the RAW model
    # (torch.compile prefixes state_dict keys with _orig_mod. if you save the wrapper).
    torch.manual_seed(0)
    data = TensorData(torch.randint(0, 64, (4000,)))
    tr = Trainer(_cfg(out_dir=str(tmp_path), max_steps=3, grad_checkpoint=True), data, data)
    stats = tr.train()
    assert stats["step"] == 3
    ck = str(tmp_path / "ck.pt")
    tr.save_checkpoint(ck)
    tr2 = Trainer(_cfg(max_steps=3), data, data)
    tr2.load_checkpoint(ck)  # keys must match the uncompiled/unwrapped model
```
(`compile=True` is exercised only in the GPU-marked test below — torch.compile on CPU in CI is slow; add:)
```python
@pytest.mark.gpu
def test_compile_flag_on_cuda(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    data = TensorData(torch.randint(0, 64, (8000,)))
    tr = Trainer(_cfg(device="cuda", dtype="bfloat16", out_dir=str(tmp_path), max_steps=3,
                      compile=True), data, data)
    assert tr.train()["step"] == 3
```

Run → FAIL (`RunConfig` has no `grad_checkpoint`). Implement:

`src/microlab/train/config.py` — after `ckpt_keep`:
```python
    grad_checkpoint: bool = False  # recompute activations backward: ~30x less act memory
    compile: bool = False          # torch.compile the model (CUDA; first step compiles)
```

`src/microlab/model/reference/variants.py` — in `VariantGPT.__init__` add `self.grad_checkpoint = False` (plain attribute); in `forward`, replace the block loop:
```python
        for i, block in enumerate(self.transformer.h):
            cache_arg = (kv_cache, i) if kv_cache is not None else None
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x, kv_cache=cache_arg)
```

`src/microlab/train/trainer.py` — in `__init__` after `self.model.to(self.device)`:
```python
        self.raw_model = self.model  # state_dict source of truth (survives torch.compile)
        self.raw_model.grad_checkpoint = cfg.grad_checkpoint
        if cfg.compile:
            self.model = torch.compile(self.model)
```
and in `save_checkpoint`/`load_checkpoint` swap `self.model.state_dict()` → `self.raw_model.state_dict()` and `self.model.load_state_dict(...)` → `self.raw_model.load_state_dict(...)`. In `_log_sample`, generation must also use `self.raw_model` (compiled graphs dislike shape churn): change `generate(self.model, ...)` to `generate(self.raw_model, ...)` and the `was_training`/`train()` calls to the raw model.

Run the trainer tests → PASS (CPU set), full suite green.

- [ ] **Step 8: DDP script** — `scripts/pretrain_ddp.py`:

```python
"""Multi-GPU data-parallel pretraining via torchrun. Reuses the single-GPU Trainer;
each rank samples its own data stream (seed+rank), gradients sync through DDP, rank 0
logs/checkpoints. Scaling-efficiency drill for the Phase 7 cloud rung.

    torchrun --nproc_per_node=4 scripts/pretrain_ddp.py configs/150m.py \
        --data-dir data/shards/tinystories
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402

from microlab.data.shard_dataset import ShardDataset  # noqa: E402
from microlab.tokenizer.fast import FastTokenizer  # noqa: E402
from microlab.train.trainer import Trainer, get_lr  # noqa: E402

from scripts.pretrain import load_config  # noqa: E402  (see note below)


class DDPTrainer(Trainer):
    """Trainer whose train_step syncs grads across ranks (no_sync on all but the last
    micro-step) and averages the logged loss over the world."""

    def __init__(self, cfg, train_ds, val_ds, tokenizer, rank: int, world: int) -> None:
        super().__init__(cfg, train_ds, val_ds, tokenizer=tokenizer)
        self.rank, self.world = rank, world
        self.data_gen = torch.Generator().manual_seed(cfg.seed + rank)  # shard by stream
        self.ddp = DDP(self.model, device_ids=[rank])

    def train_step(self) -> float:
        cfg = self.cfg
        lr = get_lr(self.step, cfg)
        self.last_lr = lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.ddp.train()
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for micro in range(cfg.grad_accum):
            x, y = self.train_data.get_batch(cfg.block_size, cfg.batch_size,
                                             self.device, self.data_gen)
            sync = micro == cfg.grad_accum - 1
            ctx = self.ddp.no_sync() if not sync else torch.enable_grad()
            with ctx, self._autocast():
                _, loss = self.ddp(x, y)
                loss = loss / cfg.grad_accum
            loss.backward()
            total += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.ddp.parameters(), cfg.grad_clip if cfg.grad_clip > 0 else float("inf"))
        self.last_grad_norm = float(grad_norm)
        self.optimizer.step()
        self.step += 1
        t = torch.tensor(total, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        return t.item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--data-dir", default="data/shards")
    args = ap.parse_args()

    rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    world = dist.get_world_size()

    cfg = load_config(args.config)
    cfg.device = f"cuda:{rank}"
    tok_path = Path(args.data_dir) / "tokenizer.json"
    tok = FastTokenizer.load(str(tok_path)) if tok_path.exists() else None
    if tok is not None:
        cfg.vocab_size = tok.vocab_size
    if rank != 0:
        cfg.log_interval = 0
        cfg.ckpt_interval = 0
        cfg.eval_interval = 0

    train_ds = ShardDataset(args.data_dir, split="train")
    val_ds = ShardDataset(args.data_dir, split="val")
    trainer = DDPTrainer(cfg, train_ds, val_ds, tok, rank, world)
    ckpts = sorted(Path(cfg.out_dir).glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if ckpts:
        trainer.load_checkpoint(str(ckpts[-1]))
        if rank == 0:
            print(f"resumed from {ckpts[-1]}")
    stats = trainer.train()
    if rank == 0:
        ppl = math.exp(stats["val_loss"]) if stats.get("val_loss") is not None else float("nan")
        print(f"done: step={stats['step']} val_loss={stats['val_loss']} ppl={ppl:.2f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```
NOTE: `from scripts.pretrain import load_config` fails (scripts isn't a package) — copy the 5-line `load_config` from `scripts/pretrain.py` into this file instead, and drop that import. This script cannot be fully tested on the single-GPU box; verify it PARSES and single-process torchrun launches to the point of `init_process_group` (`torchrun --nproc_per_node=1 scripts/pretrain_ddp.py configs/150m.py --data-dir data/shards/tinystories` run for ~60s then Ctrl-C / kill; it must reach step logs). It gets exercised for real on the rented node.

- [ ] **Step 9: Runbook** — `ops/lambda-distributed.md`:

```markdown
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
```

- [ ] **Step 10: Guide** — `docs/hand-write/phase7-distributed.md`:

```markdown
> **Exercise — on `main`, no branch switching.** Implement the stub in
> `src/microlab/exercises/phase07_distributed.py`, then run
> `pytest tests/exercises/test_phase07_distributed.py -m exercise` to grade it.

# START HERE — distributed training (Phase 7)

How labs train models that don't fit on one GPU — and the phase where the 1B capstone
happens. One hand-write: `memory_budget` — params/grads/optimizer/activations per GPU
under data/tensor/pipeline parallelism and ZeRO stages 0-3. Every "can we afford to
train X" conversation in every lab starts with this arithmetic. Graded against the
oracle across a 7B/70B config matrix.

## The three rungs

1. **Local (free):** set `grad_checkpoint=True` in a config and watch VRAM drop ~30x on
   activations while tokens/sec dips ~25%; set `compile=True` and watch tokens/sec rise.
   Measure both on the 150M config; record the table.
2. **Cloud drills (~$25-50):** `ops/lambda-distributed.md`. DDP the 150M across 1/2/4
   GPUs, measure scaling efficiency; FSDP the 1B config and check your memory_budget()
   prediction against nvidia-smi. Your closed-form exercise meets reality here.
3. **The 1B capstone:** venue decided by the vendor spike (runbook step 0). LR comes from
   Phase 4's muP transfer, the fit-check from your memory budget, the restart durability
   from the systemd + resume infrastructure already proven on the 150M run.

## Readings

Megatron-LM (tensor parallelism: split the matmuls), ZeRO (shard the optimizer states —
where the 12 bytes/param actually go). Both in the console.
```

- [ ] **Step 11: Verify + commit** — oracle + trainer tests green, exercise red-by-NotImplemented, full suite green, ruff clean. Also confirm the LIVE run is unaffected: `systemctl --user is-active microlab-train-150m` → active (the running process uses the old code; just confirm no file it re-reads changed).

```bash
git add -A && git commit -m "feat(phase7): distributed — memory-budget oracle/exercise, grad-ckpt+compile flags, DDP script, cloud runbook"
```

---

### Task 11: Phase 8 — RoPE position interpolation

**Files:**
- Modify: `src/microlab/model/reference/continued.py` (add `interpolated_rope_cache`)
- Test: additions to `tests/model/` file that covers continued (find with `grep -rln "continued" tests/model tests/`; if none besides exercises, create `tests/model/test_continued_rope.py`)
- Modify: `src/microlab/exercises/phase08_continued.py`, its exercise test file
- Modify: `docs/hand-write/phase8-continued.md`

**Interfaces:**
- Produces: `interpolated_rope_cache(seq_len: int, head_dim: int, scale: float, base: float = 10000.0) -> tuple[Tensor, Tensor]` — identical to `build_rope_cache` but positions are `t/scale` (Chen et al. position interpolation), so a model trained at block N can address N*scale positions inside its trained rotation range.

- [ ] **Step 1: Failing oracle test** — `tests/model/test_continued_rope.py`:

```python
"""Position-interpolation oracle: scaled positions must land exactly on the original
cache's rows at integer-aligned points."""

import torch

from microlab.model.reference.continued import interpolated_rope_cache
from microlab.model.reference.variants import build_rope_cache


def test_scale_one_is_identity():
    a_cos, a_sin = build_rope_cache(64, 8)
    b_cos, b_sin = interpolated_rope_cache(64, 8, scale=1.0)
    assert torch.allclose(a_cos, b_cos, atol=1e-6) and torch.allclose(a_sin, b_sin, atol=1e-6)


def test_scale_two_hits_original_positions_at_even_rows():
    base_cos, base_sin = build_rope_cache(32, 8)
    int_cos, int_sin = interpolated_rope_cache(64, 8, scale=2.0)
    assert torch.allclose(int_cos[::2], base_cos, atol=1e-5)
    assert torch.allclose(int_sin[::2], base_sin, atol=1e-5)


def test_interpolated_frequencies_stay_in_trained_range():
    cos, _ = interpolated_rope_cache(128, 8, scale=4.0)
    base_cos, _ = build_rope_cache(32, 8)
    assert cos.shape[0] == 128
    # max rotation angle at the last interpolated position == last trained position's
    assert torch.allclose(cos[-1], base_cos[-1], atol=1e-4)
```

- [ ] **Step 2: Verify failure.** **Step 3: Implement** in `continued.py` (add import `from microlab.model.reference.variants import build_rope_cache` is NOT needed — implement directly):

```python
def interpolated_rope_cache(seq_len: int, head_dim: int, scale: float, base: float = 10000.0):
    """Position-interpolated RoPE tables (Chen et al. 2023): compress positions by
    `scale` so seq_len positions fit inside the rotation range the model was trained on.
    scale=1 reproduces build_rope_cache exactly; scale=2 lets a 1024-trained model
    address 2048 positions (finetune briefly after swapping the cache in)."""
    import torch

    assert head_dim % 2 == 0
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float() / scale
    freqs = torch.outer(t, theta)
    return freqs.cos(), freqs.sin()
```
(Move the `import torch` to the module top if not already there — it is, keep style consistent and ruff-clean.)

- [ ] **Step 4: Oracle green.** **Step 5: Exercise stub** (append to `src/microlab/exercises/phase08_continued.py`):

```python
def interpolated_rope_cache(seq_len: int, head_dim: int, scale: float, base: float = 10000.0):
    """Position interpolation: build_rope_cache but with positions t/scale. Returns
    (cos, sin) of shape (seq_len, head_dim//2). Graded vs the reference."""
    raise NotImplementedError("theta as in build_rope_cache; t = arange(seq_len)/scale")
```

- [ ] **Step 6: Exercise test** (append to `tests/exercises/test_phase08_continued.py` before its `pytestmark`):

```python
def test_interpolated_rope_cache_matches_reference():
    import torch

    from microlab.exercises.phase08_continued import interpolated_rope_cache
    from microlab.model.reference.continued import interpolated_rope_cache as ref_cache
    for scale in (1.0, 2.0, 4.0):
        a_cos, a_sin = interpolated_rope_cache(64, 8, scale)
        b_cos, b_sin = ref_cache(64, 8, scale)
        assert torch.allclose(a_cos, b_cos, atol=1e-6)
        assert torch.allclose(a_sin, b_sin, atol=1e-6)
```

- [ ] **Step 7: Guide** — append to `docs/hand-write/phase8-continued.md`:

```markdown
## Long context (new)

Hand-write `interpolated_rope_cache` — RoPE position interpolation (Chen et al.): divide
positions by `scale` so 2048 positions fit in the rotation range trained at 1024. The
run-for-real: evaluate the 150M's perplexity at block 2048 raw (it degrades), swap in the
interpolated cache (`model.transformer.h[i].attn.rope_cos/sin` buffers + config.block_size)
and re-measure (better), then briefly continue-pretrain at 2048 (best). Also read the
Llama 3 report's data-annealing section — high-quality data late in pretraining is the
"midtraining" trick, and `build_replay_mix` from this phase is its mechanism in miniature.
```

- [ ] **Step 8: Verify + commit.**

```bash
git add -A && git commit -m "feat(phase8): RoPE position interpolation — oracle + exercise + long-context guide"
```

---

### Task 12: Console content + curriculum docs + deploy

**Files:**
- Modify: `site/content/phases.json` (insert new phase entries, extend readings)
- Modify: `docs/curriculum.md`, `plans/llm-lab-overview.md`
- Deploy + verify live.

**Interfaces:**
- Consumes: paper ids from Task 2 (use the ACTUAL ids printed in Task 2 Step 3 if they differ from the list).

- [ ] **Step 1: phases.json — extend existing readings.**
  - `phase-3` `readingPaperIds` append: `"fast-transformer-decoding-one-write-head-is-all-you-need"`, `"gqa-training-generalized-multi-query-transformer-models-from-multi-head-checkpoints"`.
  - `phase-4` append: `"tensor-programs-v-tuning-large-neural-networks-via-zero-shot-hyperparameter-transfer"`, `"small-scale-proxies-for-large-scale-transformer-training-instabilities"`.
  - `phase-8` (continued pretraining) append: `"extending-context-window-of-large-language-models-via-positional-interpolation"`.
  - Also update `phase-3`/`phase-4` `goal`/`summary` one-liners to mention GQA/MoE and muP respectively (keep style; e.g. phase-3 goal: "Change one architectural decision at a time — norms, positions, MLPs, attention head topology (GQA), and MoE routing — and measure loss, throughput, and behavior.").

- [ ] **Step 2: phases.json — insert the three new entries** between `phase-4` and `phase-8`:

```json
{
  "id": "phase-5",
  "title": "Phase 5: Interpretability",
  "status": "planned",
  "goal": "Open up the trained 150M model and find real structure: logit lens, attention patterns, induction heads.",
  "summary": "This phase turns 'I built it' into 'I can see what it learned' — decoding every layer's residual stream with the model's own unembedding, scoring induction heads on repeated sequences, and (stretch) watching induction heads form across saved checkpoints. Web reading: Anthropic's In-context Learning and Induction Heads (transformer-circuits.pub).",
  "tasks": [],
  "readingPaperIds": [
    "eliciting-latent-predictions-from-transformers-with-the-tuned-lens",
    "locating-and-editing-factual-associations-in-gpt"
  ]
},
{
  "id": "phase-6",
  "title": "Phase 6: Inference Engineering",
  "status": "planned",
  "goal": "Build everything between a checkpoint and a served token: KV cache, sampling, quantization, speculative decoding.",
  "summary": "This phase makes inference constraints concrete — why generation is memory-bound, why GQA exists (measure the 4x KV-cache shrink), what int8/int4 cost in perplexity, and how a draft model makes decoding faster for free. The KV-cache exercise is graded by exact token-match against uncached generation.",
  "tasks": [],
  "readingPaperIds": [
    "efficient-memory-management-for-large-language-model-serving-with-pagedattention",
    "fast-inference-from-transformers-via-speculative-decoding",
    "gptq-accurate-post-training-quantization-for-generative-pre-trained-transformers"
  ]
},
{
  "id": "phase-7",
  "title": "Phase 7: Distributed Training",
  "status": "planned",
  "goal": "Learn the parallelism vocabulary of every frontier lab — DP/TP/PP, ZeRO, FSDP — and feel it on rented multi-GPU hardware. Ends with the 1B capstone.",
  "summary": "This phase hand-writes the per-GPU memory budget, proves it against nvidia-smi on a rented 4x A100 node (~$25-50), measures DDP scaling efficiency on the real 150M training script, and opens with a vendor-affordability research spike that decides whether the 1B capstone trains in the cloud (~12-14h on 8x H100) or locally (~3-4 weeks).",
  "tasks": [],
  "readingPaperIds": [
    "megatron-lm-training-multi-billion-parameter-language-models-using-model-parallelism",
    "zero-memory-optimizations-toward-training-trillion-parameter-models"
  ]
},
```

- [ ] **Step 3: `docs/curriculum.md`** — replace the phase table with the 17-row version (rows 0–4 updated where extended, new 5/6/7, old rows shifted). New/changed rows:

```markdown
| 3 | Architecture ablations | RMSNorm, RoPE, SwiGLU, GQA, MoE routing + load-balance loss | — |
| 4 | Scaling experiments | param/FLOP count, scaling-law fit, muP transfer table | compute-optimal 1B config |
| 5 | Interpretability | logit lens, induction-head score | interp report on the 150M ckpt |
| 6 | Inference engineering | KV-cached generate, sampling zoo, groupwise quant, speculative accept | inference bench on the 150M ckpt |
| 7 | Distributed training | per-GPU memory budget (DP/TP/PP x ZeRO) | grad-ckpt/compile drills + cloud DDP + 1B capstone |
```
and update the exercise-numbering prose to `phase00…phase15`.

- [ ] **Step 4: `plans/llm-lab-overview.md`** — three edits:
  1. In **Working Assumptions**, add: `- Cloud budget: hundreds of dollars total is acceptable for multi-GPU educational runs (Phase 7 drills ~$25-50; 1B capstone ~$300-400 if the vendor spike favors cloud); thousands is not.`
  2. Insert new **Phase 5/6/7** sections after Phase 4 (goal + deliverables + key readings, 6–10 lines each, mirroring the existing section style; renumber the old Phase 5–13 headings to 8–16).
  3. Append a new top-level section:
```markdown
## Explicitly Out of Scope

Decided exclusions, not omissions: multimodality (would double the curriculum),
RAG/retrieval systems (application-layer, not model-building), and deep safety work
(red-teaming, jailbreak evaluation) beyond the Constitutional AI reading. Revisit after
Phase 16 if interest survives contact with the 1B capstone.
```

- [ ] **Step 5: Validate + deploy + live verify**

```bash
conda run -n microlab python -c "import json; json.load(open('site/content/phases.json')); print('valid')"
conda run -n microlab python -m pytest tests/ --ignore=tests/exercises -q
cd site && npm run build && cd ..
systemctl --user restart microlab-site && sleep 3 && systemctl --user is-active microlab-site
```
Then verify the LIVE authed path (per project rule: log in and check authed endpoints, not just a 200 on /login): mint a session cookie exactly as done before —
```bash
COOKIE=$(conda run -n microlab python -c "
import sys; sys.path.insert(0, 'src')
from microlab.console.app import create_app
from flask import session
app = create_app()
with app.test_request_context():
    session['authed'] = True
    print(app.session_interface.get_signing_serializer(app).dumps(dict(session)))")
curl -s -H "Cookie: session=$COOKIE" https://microlab.rje.ai/api/state | \
  conda run -n microlab python -c "
import json, sys
s = json.load(sys.stdin)
ids = [p['id'] for p in s['phases']]
assert ids == [f'phase-{i}' for i in range(17)], ids
assert len(s['papers']) == 62, len(s['papers'])
print('live: 17 phases, 62 papers')"
```

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: curriculum expansion live — 17 phases, new readings, cloud-budget + out-of-scope docs"
```

---

## Self-review notes (done during planning)

- Spec coverage: GQA (T4), MoE (T5), muP+stability (T6), interp (T7), inference (T8+T9), distributed+flags+runbook+vendor-spike (T10), long context+annealing (T11), papers (T2+T3), renumber (T1), console/docs/deploy (T12). Speculative→distillation callback lands in the phase-6 guide (T9) — phases.json for phase-14 needs no change.
- The `34x` activation multiplier and DDP loss-averaging are documented approximations; their empirical check is explicitly part of the Phase 7 cloud drill, not CI.
- Live-run safety: T4 `test_default_none_keeps_variantgpt_identical` + T8 `test_default_path_untouched` + T10 raw-model checkpointing test gate every `variants.py`/`trainer.py` touch.
- Type consistency: `KVCache.append(layer, k, v)` is the single cache mutation API used by both attention classes and `generate_cached`; `n_kv_head` name is uniform across config/tests/exercises.
