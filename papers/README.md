# Microlab Paper Library

This folder contains local PDF copies of the LLM papers referenced in the Microlab curriculum.
The canonical machine-readable list is `manifest.json`.

## Topics

### architecture

- [Fast Transformer Decoding: One Write-Head is All You Need](architecture/2019-shazeer-fast-transformer-decoding-one-write-head-is-all-you-need.pdf) (2019) - [source](https://arxiv.org/abs/1911.02150)
- [Root Mean Square Layer Normalization](architecture/2019-zhang-root-mean-square-layer-normalization.pdf) (2019) - [source](https://arxiv.org/abs/1910.07467)
- [GLU Variants Improve Transformer](architecture/2020-shazeer-glu-variants-improve-transformer.pdf) (2020) - [source](https://arxiv.org/abs/2002.05202)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](architecture/2021-su-roformer-rotary-position-embedding.pdf) (2021) - [source](https://arxiv.org/abs/2104.09864)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](architecture/2021-fedus-switch-transformers.pdf) (2021) - [source](https://arxiv.org/abs/2101.03961)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](architecture/2022-dao-flashattention.pdf) (2022) - [source](https://arxiv.org/abs/2205.14135)
- [Extending Context Window of Large Language Models via Positional Interpolation](architecture/2023-chen-extending-context-window-via-positional-interpolation.pdf) (2023) - [source](https://arxiv.org/abs/2306.15595)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](architecture/2023-ainslie-gqa-training-generalized-multi-query-transformer-models.pdf) (2023) - [source](https://arxiv.org/abs/2305.13245)

### efficient-finetuning

- [LoRA: Low-Rank Adaptation of Large Language Models](efficient-finetuning/2021-hu-lora.pdf) (2021) - [source](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](efficient-finetuning/2023-dettmers-qlora.pdf) (2023) - [source](https://arxiv.org/abs/2305.14314)

### evaluation

- [Measuring Massive Multitask Language Understanding](evaluation/2020-hendrycks-measuring-massive-multitask-language-understanding.pdf) (2020) - [source](https://arxiv.org/abs/2009.03300)
- [Evaluating Large Language Models Trained on Code](evaluation/2021-chen-evaluating-large-language-models-trained-on-code.pdf) (2021) - [source](https://arxiv.org/abs/2107.03374)
- [Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models](evaluation/2022-big-bench-authors-beyond-the-imitation-game.pdf) (2022) - [source](https://arxiv.org/abs/2206.04615)
- [Holistic Evaluation of Language Models](evaluation/2022-liang-holistic-evaluation-of-language-models.pdf) (2022) - [source](https://arxiv.org/abs/2211.09110)
- [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](evaluation/2024-chiang-chatbot-arena.pdf) (2024) - [source](https://arxiv.org/abs/2403.04132)

### foundations

- [Attention Is All You Need](foundations/2017-vaswani-attention-is-all-you-need.pdf) (2017) - [source](https://arxiv.org/abs/1706.03762)
- [Mixed Precision Training](foundations/2017-micikevicius-mixed-precision-training.pdf) (2017) - [source](https://arxiv.org/abs/1710.03740)
- [Language Models are Few-Shot Learners](foundations/2020-brown-language-models-are-few-shot-learners.pdf) (2020) - [source](https://arxiv.org/abs/2005.14165)
- [Scaling Laws for Neural Language Models](foundations/2020-kaplan-scaling-laws-for-neural-language-models.pdf) (2020) - [source](https://arxiv.org/abs/2001.08361)
- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](foundations/2022-yang-tensor-programs-v-zero-shot-hyperparameter-transfer.pdf) (2022) - [source](https://arxiv.org/abs/2203.03466)
- [Training Compute-Optimal Large Language Models](foundations/2022-hoffmann-training-compute-optimal-large-language-models.pdf) (2022) - [source](https://arxiv.org/abs/2203.15556)
- [Small-scale proxies for large-scale Transformer training instabilities](foundations/2023-wortsman-small-scale-proxies-for-training-instabilities.pdf) (2023) - [source](https://arxiv.org/abs/2309.14322)

### inference

- [Fast Inference from Transformers via Speculative Decoding](inference/2022-leviathan-fast-inference-via-speculative-decoding.pdf) (2022) - [source](https://arxiv.org/abs/2211.17192)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](inference/2022-frantar-gptq-accurate-post-training-quantization.pdf) (2022) - [source](https://arxiv.org/abs/2210.17323)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](inference/2023-kwon-efficient-memory-management-with-pagedattention.pdf) (2023) - [source](https://arxiv.org/abs/2309.06180)

### instruction-tuning

- [Finetuned Language Models Are Zero-Shot Learners](instruction-tuning/2021-wei-finetuned-language-models-are-zero-shot-learners.pdf) (2021) - [source](https://arxiv.org/abs/2109.01652)
- [Scaling Instruction-Finetuned Language Models](instruction-tuning/2022-chung-scaling-instruction-finetuned-language-models.pdf) (2022) - [source](https://arxiv.org/abs/2210.11416)
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](instruction-tuning/2022-wang-self-instruct.pdf) (2022) - [source](https://arxiv.org/abs/2212.10560)
- [Training language models to follow instructions with human feedback](instruction-tuning/2022-ouyang-instructgpt.pdf) (2022) - [source](https://arxiv.org/abs/2203.02155)

### interpretability

- [Locating and Editing Factual Associations in GPT](interpretability/2022-meng-locating-and-editing-factual-associations-in-gpt.pdf) (2022) - [source](https://arxiv.org/abs/2202.05262)
- [Eliciting Latent Predictions from Transformers with the Tuned Lens](interpretability/2023-belrose-eliciting-latent-predictions-with-the-tuned-lens.pdf) (2023) - [source](https://arxiv.org/abs/2303.08112)

### modern-llm-recipes

- [Cramming: Training a Language Model on a Single GPU in One Day](modern-llm-recipes/2022-geiping-cramming.pdf) (2022) - [source](https://arxiv.org/abs/2212.14034)
- [OPT: Open Pre-trained Transformer Language Models](modern-llm-recipes/2022-zhang-opt-open-pre-trained-transformer-language-models.pdf) (2022) - [source](https://arxiv.org/abs/2205.01068)
- [LLaMA: Open and Efficient Foundation Language Models](modern-llm-recipes/2023-touvron-llama-open-and-efficient-foundation-language-models.pdf) (2023) - [source](https://arxiv.org/abs/2302.13971)
- [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](modern-llm-recipes/2023-biderman-pythia.pdf) (2023) - [source](https://arxiv.org/abs/2304.01373)
- [DeepSeek-V3 Technical Report](modern-llm-recipes/2024-deepseek-ai-deepseek-v3-technical-report.pdf) (2024) - [source](https://arxiv.org/abs/2412.19437)
- [The Llama 3 Herd of Models](modern-llm-recipes/2024-dubey-the-llama-3-herd-of-models.pdf) (2024) - [source](https://arxiv.org/abs/2407.21783)

### preferences-rlhf

- [Proximal Policy Optimization Algorithms](preferences-rlhf/2017-schulman-proximal-policy-optimization-algorithms.pdf) (2017) - [source](https://arxiv.org/abs/1707.06347)
- [Fine-Tuning Language Models from Human Preferences](preferences-rlhf/2019-ziegler-fine-tuning-language-models-from-human-preferences.pdf) (2019) - [source](https://arxiv.org/abs/1909.08593)
- [Constitutional AI: Harmlessness from AI Feedback](preferences-rlhf/2022-bai-constitutional-ai.pdf) (2022) - [source](https://arxiv.org/abs/2212.08073)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](preferences-rlhf/2023-rafailov-direct-preference-optimization.pdf) (2023) - [source](https://arxiv.org/abs/2305.18290)
- [KTO: Model Alignment as Prospect Theoretic Optimization](preferences-rlhf/2024-ethayarajh-kto.pdf) (2024) - [source](https://arxiv.org/abs/2402.01306)
- [ORPO: Monolithic Preference Optimization without Reference Model](preferences-rlhf/2024-hong-orpo.pdf) (2024) - [source](https://arxiv.org/abs/2403.07691)

### reasoning-rl

- [Training Verifiers to Solve Math Word Problems](reasoning-rl/2021-cobbe-training-verifiers-to-solve-math-word-problems.pdf) (2021) - [source](https://arxiv.org/abs/2110.14168)
- [STaR: Bootstrapping Reasoning With Reasoning](reasoning-rl/2022-zelikman-star.pdf) (2022) - [source](https://arxiv.org/abs/2203.14465)
- [Solving math word problems with process- and outcome-based feedback](reasoning-rl/2022-uesato-process-and-outcome-feedback.pdf) (2022) - [source](https://arxiv.org/abs/2211.14275)
- [Let's Verify Step by Step](reasoning-rl/2023-lightman-lets-verify-step-by-step.pdf) (2023) - [source](https://arxiv.org/abs/2305.20050)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](reasoning-rl/2024-deepseek-ai-deepseekmath.pdf) (2024) - [source](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](reasoning-rl/2025-deepseek-ai-deepseek-r1.pdf) (2025) - [source](https://arxiv.org/abs/2501.12948)
- [s1: Simple test-time scaling](reasoning-rl/2025-muennighoff-s1-simple-test-time-scaling.pdf) (2025) - [source](https://arxiv.org/abs/2501.19393)

### systems

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](systems/2019-shoeybi-megatron-lm-model-parallelism.pdf) (2019) - [source](https://arxiv.org/abs/1909.08053)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](systems/2019-rajbhandari-zero-memory-optimizations.pdf) (2019) - [source](https://arxiv.org/abs/1910.02054)

### tokenizers-data

- [Neural Machine Translation of Rare Words with Subword Units](tokenizers-data/2015-sennrich-neural-machine-translation-of-rare-words-with-subword-units.pdf) (2015) - [source](https://arxiv.org/abs/1508.07909)
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](tokenizers-data/2018-kudo-sentencepiece.pdf) (2018) - [source](https://arxiv.org/abs/1808.06226)
- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](tokenizers-data/2020-gao-the-pile.pdf) (2020) - [source](https://arxiv.org/abs/2101.00027)
- [Deduplicating Training Data Makes Language Models Better](tokenizers-data/2021-lee-deduplicating-training-data-makes-language-models-better.pdf) (2021) - [source](https://arxiv.org/abs/2107.06499)
- [DataComp-LM: In search of the next generation of training sets for language models](tokenizers-data/2024-li-datacomp-lm.pdf) (2024) - [source](https://arxiv.org/abs/2406.11794)
- [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](tokenizers-data/2024-penedo-the-fineweb-datasets.pdf) (2024) - [source](https://arxiv.org/abs/2406.17557)

### tool-use

- [ReAct: Synergizing Reasoning and Acting in Language Models](tool-use/2022-yao-react.pdf) (2022) - [source](https://arxiv.org/abs/2210.03629)
- [Gorilla: Large Language Model Connected with Massive APIs](tool-use/2023-patil-gorilla.pdf) (2023) - [source](https://arxiv.org/abs/2305.15334)
- [ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](tool-use/2023-qin-toolllm.pdf) (2023) - [source](https://arxiv.org/abs/2307.16789)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](tool-use/2023-schick-toolformer.pdf) (2023) - [source](https://arxiv.org/abs/2302.04761)
