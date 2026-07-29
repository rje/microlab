# Is 3D asset generation a scale-light research area?

Exploratory scan run 2026-07-29, following the ancillary question about research areas less
dependent on hardware scale. **This is a scan, not a program proposal** — the lab's focus is
unchanged. Recorded because the numbers are reusable and most of them are hard to find (a majority
of 3D papers omit training compute entirely).

Companion context: JEPA does not generate; see `lit-review-design-world-models.md` for the
JEPA-as-encoder framing. The JEPA×3D-generation intersection is empty, and REPA's own ablation
measured I-JEPA as a *worse* alignment target than DINOv2 (FID 11.6 vs 9.7), so that hole is
unclaimed partly because the ingredient lost.

---

## Answer: yes, and the reason is the floor, not the ceiling

Frontier open 3D generation costs about the same as a small LLM pretrain. **What differs is the
minimum viable result.**

| | 3D generation | LLM |
|---|---|---|
| Frontier open model | CLAY-XL 3,840 GPU-days; TripoSG 3,744 GPU-days | Llama-3.2-1B 15,417 H100-days; TinyLlama-1.1B 1,440 A100-days |
| Citable paper floor | **8–130 GPU-days** (iFlame 8, MeshGPT 24, MeshAnything 36, Direct3D-S2 128) | 24 GPU-days buys a ~150M model nobody uses |

Frontier 3D is ~2.6× a TinyLlama and ~1/10 of Llama-3.2-1B. The ceiling is comparable. But in 3D a
24-GPU-day run is a CVPR/NeurIPS/ICCV paper, and in language it is a toy. **That asymmetry is the
entire scale-light case.**

## Parameter scaling in the incumbent architecture is nearly flat

The VecSet family (3DShape2VecSet → Michelangelo → CraftsMan → CLAY → TripoSG → Step1X-3D →
Hunyuan3D) is the dominant lineage. LATTICE, Appendix A.1, verbatim: *"VecSet models hardly benefit
from an increased number of parameters… the three results are largely similar, with Hunyuan3D-2-mini
(0.6B) even showing slightly better performance."* That compares 0.6B / 1.1B / 3B — a 5× increase
across the leading production lineage buying nothing, with the *smallest* arguably best.

Corroboration from TRELLIS Table 5: 342M → 2B moves FD_dinov2 121.45 → 93.96 but CLIP alignment only
25.41 → 25.71 — flat on semantics for 6× the parameters.

*Caveat:* the Hunyuan checkpoints are separate releases with differing training data, not a
controlled fixed-data sweep. Suggestive, not decisive. LATTICE's own thesis is that **localizability,
not parameters, unlocks scaling** — *"larger models are useful only when there is a clear
correspondence between conditions and outputs"* — and they concede the flatness *"could also be
explained by the lack of large-scale 3D data."*

## Capability is bought with compression, not parameters

The two cleanest results, and they point the same way:

- **BPT** (arXiv 2411.07025, Tencent Hunyuan) — a mesh tokenizer at 0.26 compression ratio (vs
  MeshAnythingV2 0.46, EdgeRunner 0.47), letting a **500M** model generate meshes exceeding **8,000
  faces** in a 9,600-token window, against LLaMA-Mesh's **8B** params capped at **500 faces**. A 16×
  parameter disadvantage producing a 16× capability advantage, purely from sequence compression.
  Trained on 32×L40 × ~7 d. Evidence strong: all figures stated, and the paper's own limitations
  section calls 500M "still insufficient" rather than overclaiming. Caveat: face capability and
  generation quality are different axes; no head-to-head benchmark against LLaMA-Mesh.
- **COD-VAE** (arXiv 2503.08737, **ICCV 2025**) — 64 latent vectors vs VecSet's 512–1024 (16×
  compression), **20.8× faster generation while improving quality**: ShapeNet IoU 96.5% vs
  VecSet-512's 96.2%, Rendering-FID 37.05 vs 44.18. Trained on **4×RTX 4090 × ~3 days**. Evidence
  strong: peer-reviewed, controlled same-dataset comparison against the incumbent representation.
  Parameter counts unpublished.

## What one RTX 6000 Ada (48GB) can actually train

Conversion basis: RTX 6000 Ada ≈ 0.5× A100 in practice (960 GB/s vs 1,555–2,039; ~182 TFLOPS dense
FP16 vs 312 BF16; no NVLink). So 1 A100-day ≈ 2 RTX 6000 Ada-days.

**Reachable (≤ ~45 days):**

| Result | Published cost | On one card |
|---|---|---|
| Objaverse++ curation experiment (50K curated beat 100K random, 83.5% of votes) | 8×H100 × 6 h | ~6 d |
| COD-VAE ShapeNet VAE (ICCV 2025) | 4×RTX 4090 × 3 d | ~12–14 d |
| **Michelangelo** — still a benchmarked baseline in Feb 2026 | 8×V100 × 5 d | ~15–25 d |
| **iFlame** — repo describes it as *"a single-GPU trainable unconditional mesh generative model"* | 4 GPUs × 2 d | ~16–32 d |
| MeshGPT VQ-VAE alone | 2×A100 × 2 d | ~8 d |
| MeshGPT full | 24 A100-d | ~48 d (marginal) |

**Not reachable:** MeshXL-125M ~5 mo · Direct3D-S2 full ~8.5 mo · GS-LRM ~1 yr · LRM ~2 yr ·
MeshXL-1.3B ~5 yr · **TripoSG / CLAY-XL ~20 yr**.

**VRAM is not the binding constraint — wall-clock is.** 48GB holds inference for every open model
(TRELLIS's stated floor is 16GB) and trains a 300M–1B DiT with gradient checkpointing. The real
limits are attention over 2,048–4,096 vecset tokens (or 45,904 sparse-voxel latents at 1024³) and
batch size — TripoSG used batch 8–16 *per GPU* across 160 GPUs.

**Is 8×A100 the floor? Only for a shippable generator.** For representation/tokenizer/VAE and
data-curation research, 1–4 GPUs is demonstrably sufficient and is where several 2025 ICCV/NeurIPS
papers came from (COD-VAE 4×4090, iFlame 4 GPUs, MeshGPT 2–4 A100).

## The actual blocker is data access and preprocessing, not FLOPs

Measured rather than quoted (probe dated 2026-07-29):

| Objaverse-XL | Count |
|---|---|
| Published headline | 10.2M |
| Public metadata rows | 9,767,011 |
| Still retrievable | ~8.73M (89.4%) — ~1.0M lost to link rot, almost all deleted/private GitHub repos |
| Reliably textured | **869,438 (8.9%)** — Thingiverse contributes 3,732,204 STL print models with no appearance data |
| Textured **and** permissively licensed | **704,807 (7.2%)** — 73.1% of GitHub rows carry license `None` |
| Production-ready (PBR + topology + UV) | 5K–15K |

Zero123-XL — Objaverse-XL's own model — trained on a 1.3M "high-quality subset." **The dataset's
authors discarded 87% of it.** Effective 3D-vs-2D data gap is ~10⁴×, not the ~570× raw counts imply.
ShapeNet has degraded to gated manual approval (shapenet.org returns HTTP 503).

Preprocessing capex is the real expense: TRELLIS-scale means watertighting, SDF/occupancy sampling,
and **150 renders per asset** across ~500K meshes (~75M renders), on a corpus where the authors' own
pipeline rendered only 54% of inputs successfully. Storage, CPU-days and dataset-approval latency are
first-class costs.

**Escape hatch:** LRM-Zero / Zeroverse (arXiv 2406.09371) trains entirely on procedurally
synthesized data — primitives with random texturing, height fields, boolean differences, wireframes
— with *zero* real 3D assets, reporting competitiveness with Objaverse-trained models. If it holds it
removes the licensing ceiling, link rot, gating, and preprocessing capex at once. ⚠️ The reported
object count and PSNR shortfall are second-hand; the agent reached only the abstract. **Needs a
direct read before anyone leans on it.**

## House-rule check: geometry is clean, appearance is not

Relevant to `build-capability-not-distill`.

**Native-3D, no 2D generative prior** — trainable from scratch on 3D data: 3DShape2VecSet,
Michelangelo, TRELLIS/TRELLIS.2, TripoSG, Direct3D(-S2), Sparc3D, LATTICE, CLAY *(geometry stage)*,
Step1X-3D *(geometry stage)*; AR mesh: MeshGPT, MeshAnything V1/V2, MeshXL, BPT, iFlame;
feed-forward: LRM (*"does not rely on any guidance from pre-trained vision-language contrastive or
generative models"*), GS-LRM.

A distinction papers blur: TRELLIS uses DINOv2, LRM uses DINO ViT-B/16, Michelangelo uses frozen
CLIP. Those are frozen 2D *encoders* for conditioning — the generator trains from scratch on 3D. Only
LLaMA-Mesh borrows generative weights, and from an LLM.

**But texture/PBR is entirely 2D-prior-driven and nobody trains it from 3D alone.** Hunyuan3D-Paint
initializes from Stable Diffusion 2.1; CLAY's texture stage modifies MVDream; Step1X-3D's is SDXL +
MV-Adapter. The 5K–15K production-ready PBR objects are why. **Geometry can respect the house rule;
appearance currently cannot.**

Quality verdict favors native-3D anyway — HY3D-Bench (Feb 2026) Uni3D-I: TRELLIS 0.3641, Hunyuan3D
2.1 0.3636, CraftsMan 0.3351, Michelangelo 0.3169; every top performer is native-3D latent
diffusion. The 2026 survey calls SDS/DreamFusion **legacy**.

## Evidence-quality notes

Every model competitive *as a product* — TRELLIS, all Hunyuan3D versions, Step1X-3D, Direct3D —
either omits training duration or omits compute entirely. **Only CLAY and TripoSG can be anchored
on.** Papers verified as omitting compute outright: Direct3D 2024, OpenLRM, Shap-E/Point-E,
Hunyuan3D 2.0, Hunyuan3D 2.5.

Corrections the sweep made to its own earlier claims, recorded so they don't get re-propagated:
TripoSG's 180K→2M data ablation is **confounded** (that row also jumps 975M→4B params); ABO is
**7,953** models, not the widely-cited 8,222 (that's a 360°-image product count); Objaverse++ is
ICCV 2025 *Workshop*, not CVPR. Contested: Hunyuan3D 2.1 is **1,238M** per Tencent's own HY3D-Bench
but **3B** per LATTICE — unresolved, and it affects the VecSet scaling comparison above. Tripo's
marketed "40M assets" appears only in PR, never a paper.

## If this were ever pursued

The scale-light thesis is real but narrow: it holds for **representation and tokenizer design**,
which is exactly where the field's wins currently are (BPT, COD-VAE, LATTICE's localizability
argument), and where one card is sufficient. It does not hold for shipping a competitive image-to-3D
generator. The first thing to check would be Zeroverse — if procedural data works, the data-access
gate that actually blocks a single-GPU lab disappears.
