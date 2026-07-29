# 3D generation: mesh topology and part separability

Scan run 2026-07-29, following the design requirements "superior topology" and "submesh
separability." Companion to `scale-light-3d-generation.md` (compute/data economics).

**Status: partial.** Several sweeps died on API session limits mid-run. What follows is from the
threads that completed with primary-source verification; gaps are marked **[NOT COVERED]** rather
than filled by inference. Two claims from my earlier verbal summary are **corrected** below — see
"Corrections."

---

## Headline: the field's central topology claim is essentially unmeasured

Artist-mesh (AM) generation exists because marching-cubes extraction produces dense unusable
connectivity. That is the entire motivation for the line. **Almost nobody measures whether it
works.**

| | MeshXL | LLaMA-Mesh | MeshCraft | Nautilus |
|---|---|---|---|---|
| arXiv | 2405.20853 | 2411.09595 | 2503.23022 | 2501.14317 |
| Venue | NeurIPS 2024 | preprint only | preprint only | **ICCV 2025** |
| Max faces | 800 | 500 | 800 | **5,000** |
| Tokens/face | 9 | variable BPE | **1** (8-dim continuous VAE) | ~2.5 (0.275 × 9N) |
| Geometry metrics | COV/MMD/1-NNA/JSD/FID/KID | **none at all** | same + tri-acc/L2 | CD / HD |
| **Topology metrics** | **NONE** | **NONE** | **NONE** | **YES** |
| Output | triangle | unspecified | triangle (explicit) | triangle |

- **LLaMA-Mesh reports no 3D quantitative metrics whatsoever** — no Chamfer, no FID, nothing. Its
  claim of "mesh generation quality on par with models trained from scratch" rests on one
  qualitative figure. The words *manifold*, *watertight*, *self-intersection* never appear.
- **MeshCraft** opens by declaring that "achieving optimal mesh topology has long been a pursuit for
  3D artists" and then measures zero topology properties.
- **Nautilus is the first to quantify it, and says so**: *"Since no such metric is included in prior
  works, we adopt two traditional measures…"* Its Appendix table — Surface Holes 4.2% (vs 40.6–45.0%
  for MeshAnything V1/V2), Intersecting Faces 9.6% (vs 16.0–16.8%), Manifold **83.6%** (vs
  23.8–29.0%). Caveats: the "Missing Parts" column is **manual human inspection**, the Manifold
  column's algorithm is unstated, and its own limitations section concedes it "does not strictly
  guarantee fully manifold meshes" with intersecting-face gains "relatively less pronounced."
- **No paper in the set compares against an SDF/marching-cubes pipeline on topology metrics.**
  Nautilus even *derives its input point clouds* from those pipelines and still never benchmarks
  against them on the property it exists to improve.
- **All four are triangle-only. No quad ratio is reported anywhere**, despite "artist-created mesh"
  implying quad-dominant topology.

**So the gap is benchmark-shaped**, not model-shaped: there is no standard, automated, reproducible
topology metric suite, and the one paper that tried invented its own and used manual inspection for
part of it. That is cheap for a small lab to build and would be immediately useful to everyone in
the line.

## Part separability: crowded, and the papers state the mechanism clearly

This is **not** open territory — there was a visible 2025–2026 wave (PartGen, PartCrafter, HoloPart,
PartPacker, OmniPart, X-Part, MMPart, CoPart, AutoPartGen, FullPart, DreamPartGen, CubePart). But
they converge on *why* post-hoc segmentation cannot substitute, with quotable primary-source
statements:

- **HoloPart** (2504.07943): *"Existing 3D part segmentation methods only identify visible surface
  patches, limiting their utility."*
- **PartGen** (2412.18608): *"if the object is generated or scanned, different parts are usually
  'fused' together, missing the internal surfaces and the part boundaries."* And the reason
  generation is required: *"in extreme cases, it can hallucinate entirely invisible parts."*
- **OmniPart** (2507.06165): *"these pipelines inherently lack access to occluded or interior
  structures, leading to incomplete and view-biased segmentations."*
- **PartPacker** (2506.09980): *"any mistake in the segmentation stage can negatively impact the
  final generation quality"*, plus inference time scaling linearly in part count.
- **PartCrafter** (2506.05573): the segment-then-reconstruct pipeline *"suffers from errors in
  segmentation, extensive computational costs… and difficulties in scaling up."*

**Sampling-bias caveat, flagged by the agent:** every paper surveyed is either a generation paper
(with an incentive to critique segmentation) or a segmentation paper that never engages the
sufficiency question. Absence of a counter-argument here is weak evidence.

The segmentation zoo (SATR, PartSLIP/++, PartSTAD, ZeroPS, SAMesh, Find3D, Point-SAM, Search3D) has
a documented **generalization crisis**: PartSLIP++ scores 60.8 mIoU on its home benchmark but
**6.03–10.43 mIoU** under Find3D's independent out-of-domain test; SATR is strong on FAUST but hits
12.3 mIoU under SAMPart3D's protocol. Recurring admitted failure: methods that lift 2D foundation
models are capped by SAM/GLIP quality and are blind to occluded and interior points by construction.
Notably, **SAMesh loses to ShapeDiam — a classical non-learned baseline — on flat/mechanical
shapes**, which rhymes with the tree-sitter and PCA-probe findings elsewhere in this lab's reviews.

**Does any method achieve good topology AND genuine part separability simultaneously? No evidence
found.** That intersection is the interesting one, and it is where I would look first.

## Corrections to my earlier verbal claims

1. **"Construction programs give part separability for free" — WRONG as stated for published
   models.** The B-rep generation literature is explicitly single-body. **BrepGen** (SIGGRAPH 2024)
   says it twice: multi-body models are *deleted from its training set* (*"made from multiple bodies
   are removed"*) and *"BrepGen supports only single body solids; more complicated CAD models with
   multiple assembled bodies are left to future work."* **SolidGen** (TMLR 2023) targets *"a single
   solid model."* **HoLa** (SIGGRAPH 2025) never addresses it and sews all surfaces into one
   watertight model. The claim holds for CAD *as a format* but **no published generative model does
   multi-body**. That is a genuine gap — and it is exactly the "submesh separability" requirement.
2. **"B-rep is motivated by avoiding marching-cubes soup" — not supported.** All three papers were
   searched for *marching cubes*, *implicit*, *SDF*, *iso-surface*, *mesh soup*. **None makes that
   argument.** Their stated motivation is uniformly: CAD software consumes B-rep, parametric surfaces
   beat planar facets, and editability. If that argument is needed, it must be sourced elsewhere.

## CAD program generation: what's real

- **Validity is worse than commonly cited.** DeepCAD's famous ~3.3% invalid ratio is an
  **autoencoding** number. Third-party measurement (CAD-GPT, 2412.19663) puts *generation-time*
  invalidity at **DeepCAD 23.16%, SkexGen 22.32%, HNC-CAD 18.64%, GPT-4 64.37%** — roughly 7× worse
  than the number people quote. **SkexGen and HNC-CAD report no validity metric at all**; HNC-CAD
  names invalidity as its primary failure mode and notes its losses "do not explicitly penalize
  invalid geometries."
- **Cross-paper metrics are unsound.** DeepCAD, SkexGen and HNC-CAD report three different values for
  the same baseline under the same metric names (DeepCAD's own COV 78.13 / SkexGen reports 76.8 /
  HNC-CAD reports 80.62), with inconsistent scale factors and sample counts. Never combine rows
  across these papers.
- **Parameter counts are unpublished** for DeepCAD, SkexGen and HNC-CAD; where published in this
  lineage they are tiny (CurveGen 2.16M, TurtleGen 2.69M, JoinABLe 1.3M). GPU-hours are unpublished
  almost everywhere.
- **All B-rep methods depend on OpenCascade** for the final sew — none produces a solid without a
  kernel call. Which means the kernel is available as an exact verifier, the thing that matters for
  our purposes.
- **Surface-type split worth knowing:** SolidGen emits analytic primitives (plane/cylinder/cone/
  sphere/torus, explicitly no splines); BrepGen and HoLa fit *everything* to B-splines. The newer,
  higher-validity methods **lose analytic primitive identity** in their output.

## The live front, and the gap that fits this lab

LLM-writes-shape-program is the most active area, and it has a clean hole in the middle.

- **MeshCoder** (2508.14879, Aug 2025) — **Llama-3.2-1B + LoRA** generating Blender Python from point
  clouds, ~1M object-code pairs plus ~10M synthetic part-level pairs. Voxel IoU **86.75%** vs PLAD
  67.62% and Shape2Prog 45.03%. *Confounded:* baselines are 2019/2022 self-supervised models trained
  on ~10⁴ shapes without paired supervision; MeshCoder uses 10⁶ supervised pairs and a richer DSL.
- **3DCodeBench** (2606.01057, Google + USC) — 212 categories, **12,720 factory instances** with
  per-instance code, renders, baked GLBs and captions, explicitly positioned for SFT. Evaluates 12
  frontier VLMs. **Key finding: failures are dominated by API mismatches, not geometric reasoning —
  multi-turn error feedback lifts executability 70.2% → 97.4%.** But even executable output shows
  "disconnected or floating geometric components." Image-similarity correlates with human preference
  at r=0.964.
- **CADCodeVerify** (ICLR 2025) — VLM generates CAD code *and* answers its own validation questions
  on renders, then corrects. The closest published thing to execute-and-verify.
- **DI-PCG** (2412.15200) — the counterweight: **7.6M params, 30 GPU-hours** recovers procedural
  generator parameters via diffusion, beating LLM prompting by orders of magnitude on cost. **LLMs
  only win when the program *structure* is unknown.**

**The gap, stated by the sweep:** the 2018–2023 work trained small (1–10M param) bespoke models with
clever self-supervision on ~10⁴ shapes; the 2024–2026 work either prompts a frontier API model or
LoRA-tunes a 1B on 10⁶ synthetic pairs. **Nobody has trained a small-to-mid model from scratch on
shape programs at modern data scale.** That is precisely this lab's size class, and the verifier
(a CAD kernel or Blender) is exact and free.

Dead threads worth not re-entering: pure neural CSG-tree inference peaked at D²CSG (NeurIPS 2023) and
has no 2025–26 successor; the ShapeAssembly DSL survives only as a benchmark others compare against.

## [NOT COVERED] — sweeps that died before reporting

- **Field-guided / cross-field quad meshing as a generative target.** This was my most promising
  hypothesis (predict a curvature-aligned direction field, extract quads with a deterministic solver)
  and it remains **unverified in both directions** — I do not know whether it exists.
- Learned retopology (dense mesh → artist mesh) and whether paired training data exists at scale.
- Subdivision-surface / control-cage generation.
- Whether a topology benchmark exists beyond Nautilus's ad-hoc table. One partial signal: a paper
  reportedly claims no established topology metrics exist — **unverified, do not cite**.

## Assessment

The two requirements split cleanly on how occupied they are. **Part separability is crowded** — a
dozen 2025–26 papers, with the segmentation-is-insufficient argument well documented. **Topology is
barely measured**, which makes it both the real opening and the reason nobody can currently claim
progress on it. The intersection — clean topology *and* genuine separability — has no demonstrated
instance, and multi-body B-rep generation is explicitly future work in the strongest paper in that
line.

For a lab this size, the cheapest defensible contribution is not a better generator: it is the
**automated topology + separability metric suite** that the entire field is missing, applied first to
the existing open models. That is days of work, needs no GPU, and would tell us whether the "artist
mesh beats marching cubes" premise is even true before anyone builds on it.
