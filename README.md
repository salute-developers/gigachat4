## [HF link](https://huggingface.co/collections/ai-sage/gigachat-4)

# GigaChat 4

GigaChat 4 is a family of hybrid Mixture-of-Experts (MoE) language models combining **Gated DeltaNet (GDN)** and **DeepSeek Sparse Attention (DSA)**. The models support multilingual conversation, mathematical reasoning, coding, structured outputs, tool calling, multi-step agentic tasks, and question answering over long documents.

The final instruction-tuned models support **instant and reasoning modes within a single checkpoint**, selectable per request, as well as speculative decoding with recurrent MTP and DFlash2. The final base and instruction-tuned models feature a **1,048,576-token context window**.

The family includes two model sizes:

- **GigaChat 4 708B-A41B** — the flagship model, with 708B total parameters and 41B active per token.
- **GigaChat 4 37B-A4B** — the smaller model, with 37B total parameters and 4B active per token.

**The two model sizes are released under different licenses:** 708B under the MIT License and 37B under the Noncommercial License for Individuals 1.0. See [License](#license) for details.

[Technical Report](https://github.com/salute-developers/gigachat4/blob/main/GigaChat_4.pdf) · [Blog](https://giga.chat/blog/en) · [SGLang Cookbook](https://docs.sglang.io/cookbook/autoregressive/GigaChat/GigaChat4)

## Contents

- [Model and checkpoint list](#model-and-checkpoint-list)
- [Shared architecture](#shared-architecture)
- [GigaChat 4 708B-A41B](#gigachat-4-708b-a41b-1)
- [GigaChat 4 37B-A4B](#gigachat-4-37b-a4b-1)
- [Training and alignment](#training-and-alignment)
- [Inference and deployment](#inference-and-deployment)
  - [SGLang](#sglang)
    - [Installation](#sglang-installation)
    - [GigaChat 4 708B-A41B: FP8](#gigachat-4-708b-a41b-fp8)
    - [GigaChat 4 37B-A4B: FP8](#gigachat-4-37b-a4b-fp8)
    - [BF16, base, and intermediate checkpoints](#bf16-base-and-intermediate-checkpoints)
    - [Instant and reasoning modes](#instant-and-reasoning-modes)
  - [vLLM](#vllm)
    - [Installation](#vllm-installation)
    - [GigaChat 4 708B-A41B: FP8](#gigachat-4-708b-a41b-fp8-1)
    - [GigaChat 4 37B-A4B: FP8](#gigachat-4-37b-a4b-fp8-1)
    - [BF16 models](#bf16-models)
    - [Instant and reasoning modes](#instant-and-reasoning-modes-1)
    - [Tool calling](#tool-calling)
  - [llama.cpp](#llamacpp)
    - [Installation](#llamacpp-installation)
    - [Download GGUF weights](#download-gguf-weights)
    - [Run the server](#run-the-server)
- [Recommended agent harnesses](#recommended-agent-harnesses)
  - [Pi](#pi)
  - [Codex CLI](#codex-cli)
  - [DeepSeek Harness](#deepseek-harness)
- [License](#license)

---

## Model and checkpoint list

### GigaChat 4 708B-A41B

| Repository | Weights | Training stage |
| --- | --- | --- |
| [GigaChat4-708B-A41B](https://huggingface.co/ai-sage/GigaChat4-708B-A41B) | FP8 | Instruction-tuned |
| [GigaChat4-708B-A41B-GGUF](https://huggingface.co/ai-sage/GigaChat4-708B-A41B-GGUF) | GGUF | Instruction-tuned |
| [GigaChat4-708B-A41B-bf16](https://huggingface.co/ai-sage/GigaChat4-708B-A41B-bf16) | BF16 | Instruction-tuned |
| [GigaChat4-708B-A41B-base](https://huggingface.co/ai-sage/GigaChat4-708B-A41B-base) | BF16 | Base |
| [GigaChat4-708B-A41B-checkpoints](https://huggingface.co/ai-sage/GigaChat4-708B-A41B-checkpoints) | BF16 | Base, intermediate checkpoints |

### GigaChat 4 37B-A4B

| Repository | Weights | Training stage |
| --- | --- | --- |
| [GigaChat4-37B-A4B](https://huggingface.co/ai-sage/GigaChat4-37B-A4B) | FP8 | Instruction-tuned |
| [GigaChat4-37B-A4B-GGUF](https://huggingface.co/ai-sage/GigaChat4-37B-A4B-GGUF) | GGUF | Instruction-tuned |
| [GigaChat4-37B-A4B-bf16](https://huggingface.co/ai-sage/GigaChat4-37B-A4B-bf16) | BF16 | Instruction-tuned |
| [GigaChat4-37B-A4B-base](https://huggingface.co/ai-sage/GigaChat4-37B-A4B-base) | BF16 | Base |
| [GigaChat4-37B-A4B-checkpoints](https://huggingface.co/ai-sage/GigaChat4-37B-A4B-checkpoints) | BF16 | Base, intermediate checkpoints |

The **FP8** and **BF16** instruction-tuned repositories provide the final conversational models. **Base** models are intended for raw text completion. The **checkpoints** repositories contain intermediate training revisions; select a branch to load the desired checkpoint.

---

## Shared architecture

Both models use a hybrid MoE backbone that combines GatedDeltaNet linear-attention layers with DSA in multi-head latent attention (MLA) layers.

| Property | GigaChat 4 708B-A41B | GigaChat 4 37B-A4B |
| --- | --- | --- |
| Total / active parameters per token | 708B / 41B | 37B / 4B |
| Decoder layers / hidden size | 64 / 7,168 | 32 / 2,560 |
| Linear-attention / sparse-attention layers | 48 / 16 | 24 / 8 |
| Routed experts / active routed experts / shared experts | 256 / 8 / 1 | 192 / 8 / 1 |
| Initial decoder layers with dense FFNs | 3 | 1 |
| Maximum context length (final base and instruct models) | 1,048,576 tokens | 1,048,576 tokens |
| DSA tokens selected per query | 2,048 | 2,048 |
| Built-in MTP layers | 1, reused recurrently | 1, reused recurrently |

Intermediate checkpoints can have shorter context windows and different attention settings. Check the selected revision's `config.json` when using a training checkpoint.

### Hybrid attention: GatedDeltaNet and DSA

GatedDeltaNet and DSA layers are interleaved in a **3:1 ratio**:

- **GatedDeltaNet** maintains a fixed-size recurrent state for efficient sequence processing.
- **DSA** selectively attends to 2,048 tokens from the preceding context per query, providing access to information across long sequences.

The long-context training recipe extends the context window from 32K to 128K and then to 1M tokens. DSA is introduced through indexer-only training at 128K, followed by joint optimization of the indexer and backbone at 1M tokens.

### Mixture-of-Experts and GatedNorm

Each decoder layer combines an attention block with a feed-forward block. In MoE layers, the 708B model selects 8 of 256 routed experts per token, while the 37B model selects 8 of 192. Both models also use one shared expert. GatedNorm is applied to block outputs before residual addition.

### Speculative decoding

Both models support two approaches to speculative decoding:

- **Recurrent MTP** — an EAGLE-3-style drafter that reuses a single prediction layer across draft steps.
- **DFlash2** — a separate five-layer BF16 drafter that predicts tokens in parallel using blocks of eight positions: one anchor token and seven draft positions. Each model uses its own trained drafter.

In both approaches, the target model verifies multiple draft tokens in a single forward pass. For deployment examples, see the [GigaChat 4 SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/GigaChat/GigaChat4), which covers the 708B model.

---

## GigaChat 4 708B-A41B

GigaChat 4 708B-A41B is the flagship model of the family, designed for multilingual assistant workloads, coding, mathematical reasoning, tool use, and long-context agentic tasks.

Key highlights:

- **708B total / 41B active parameters per token**
- **64 decoder layers**, including 48 GatedDeltaNet and 16 DSA layers
- **1M-token context window** in the final base and instruction-tuned models
- **Instant and reasoning modes** available within a single instruction-tuned checkpoint
- **FP8, BF16, and GGUF** instruction-tuned weights, plus base and intermediate pretraining checkpoints

### Intermediate checkpoints

The 708B checkpoints repository includes the following training revisions:

| Stage | Step | Branch | Configured context |
| --- | --- | --- | --- |
| Pretraining | 80k | [pretrain-80k-step](https://huggingface.co/ai-sage/GigaChat4-708B-A41B-checkpoints/tree/pretrain-80k-step) | 4,096 |
| Pretraining | 120k | [pretrain-120k-step](https://huggingface.co/ai-sage/GigaChat4-708B-A41B-checkpoints/tree/pretrain-120k-step) | 4,096 |
| Pretraining | 173k | [pretrain-173k-step](https://huggingface.co/ai-sage/GigaChat4-708B-A41B-checkpoints/tree/pretrain-173k-step) | 4,096 |
| Cosine annealing | 235k | [cosine-anneal](https://huggingface.co/ai-sage/GigaChat4-708B-A41B-checkpoints/tree/cosine-anneal) | 4,096 |
| Midtraining | 250k | [midtrain-final](https://huggingface.co/ai-sage/GigaChat4-708B-A41B-checkpoints/tree/midtrain-final) | 32,768 |

The 1M-token context is available in the final [708B base model](https://huggingface.co/ai-sage/GigaChat4-708B-A41B-base) and the final instruction-tuned models. The intermediate revisions listed above have 4K or 32K configured context lengths.

### Benchmarks

The following tables summarize the performance of the final instruction-tuned GigaChat 4 708B-A41B model. Scores are reported as percentages unless otherwise specified. A dash indicates an unreported result.

#### Reasoning

##### Code

| Benchmark | GigaChat 4 708B | GigaChat 3.5 Ultra 432B | DeepSeek V4 Flash 0731 | GLM-5.3 | DeepSeek 4.1 1017 | Qwen 3.5 397B | Mistral 4 Large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LiveCodeBench v6 | 88.82 | 85.40 | 90.00 | 82.67 | 90.14 | 86.64 | 84.64 |
| SWE-bench Verified (7-hour timeout) | 80.89 | 64.70 | 81.80 | 86.67 | 79.40 | 73.00 | 79.60 |
| SWE-bench Multilingual (7-hour timeout) | 77.00 | — | 73.00 | 80.33 | 79.33 | 66.33 | 81.67 |
| DeepSWE (7-hour timeout) | 35.40 | — | 53.10 | 67.26 | 74.34 | 3.54 | 61.95 |
| NL2Repo (Cheating adjusted) (7-hour timeout) | 30.89 | 5.00 | 33.13 | 28.69 | 5.90 | 18.59 | 27.88 |
| Terminal-Bench 2.0 (7-hour timeout) | 71.26 | 30.30 | 55.01 | 71.67 | 61.80 | 56.18 | 77.78 |
| Terminal-Bench 2.1 (7-hour timeout) | 69.66 | — | 78.65 | 88.70 | 87.64 | 57.30 | 83.33 |

##### General

| Benchmark | GigaChat 4 708B | GigaChat 3.5 Ultra 432B | DeepSeek V4 Flash 0731 | GLM-5.3 | DeepSeek 4.1 1017 | Qwen 3.5 397B | Mistral 4 Large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IFBench (prompt-level loose) | 80.33 | 77.00 | 63.00 | 55.67 | 75.67 | 74.33 | 71.00 |
| JSONSchemaBench | 99.28 | 93.01 | 94.00 | 91.64 | 99.31 | 96.26 | 97.38 |
| StructEval Text | 87.64 | 85.00 | 88.00 | 85.36 | 85.36 | 85.48 | 86.10 |
| POLLUX | 81.46 | 75.50 | 70.00 | 77.96 | 78.64 | 74.16 | 73.18 |
| GraphWalks | 74.92 | 38.00 | 72.70 | 78.35 | 79.09 | 61.01 | 78.45 |
| Natural Plan | 94.53 | 80.19 | 91.03 | 80.00 | 93.73 | 92.25 | 94.39 |
| NIH Complex | 100.00 | 99.16 | — | 96.89 | 96.53 | 98.74 | 99.75 |
| AA-LCR | 70.33 | 25.00 | — | 66.70 | — | 68.00 | 73.67 |

##### STEM

| Benchmark | GigaChat 4 708B | GigaChat 3.5 Ultra 432B | DeepSeek V4 Flash 0731 | GLM-5.3 | DeepSeek 4.1 1017 | Qwen 3.5 397B | Mistral 4 Large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPQA Diamond | 82.83 | 82.30 | 91.00 | 90.40 | 90.91 | 85.35 | 87.37 |
| HMMT (pass@8) | 98.33 | 93.00 | 100.00 | 100.00 | 100.00 | 98.33 | 100.00 |
| HMMT (mean@8) | 93.75 | — | 98.75 | 92.08 | 97.92 | 82.71 | 99.58 |
| AIME 2025 (pass@8) | 96.67 | 93.00 | 96.67 | 100.00 | 100.00 | 96.67 | 100.00<sup>a</sup> |
| AIME 2025 (mean@8) | 93.33 | 89.00 | 90.00 | 93.125 | 97.60 | 93.75 | 97.08<sup>a</sup> |
| AIME 2026 (pass@8) | 96.67 | 97.00 | 100.00 | 100.00 | 100.00 | — | 100.00<sup>a</sup> |
| AIME 2026 (mean@8) | 95.00 | 91.00 | 95.00 | 95.104 | 98.23 | — | 99.17<sup>a</sup> |
| IMO-AnswerBench | 82.69 | 73.00 | 89.80 | 88.50 | 92.69 | 83.22 | 78.20 |
| HLE (text only, without tools) | 23.91 | 13.16 | 37.80 | 39.62 | 43.42 | 28.36 | 30.40 |

<sup>a</sup> Evaluated without tools.

##### Agentic

| Benchmark | GigaChat 4 708B | GigaChat 3.5 Ultra 432B | DeepSeek V4 Flash 0731 | GLM-5.3 | DeepSeek 4.1 1017 | Qwen 3.5 397B | Mistral 4 Large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BFCL v4 | 79.55 | 58.59 | 73.12 | 68.29 | 72.18 | 69.10 | 70.68 |
| τ²-Bench (mean@3) | 83.31 | 61.58 | 83.30 | 78.70 | 79.43 | 78.83 | 80.60 |
| τ³-Bench Banking (mean@3) | 18.03 | 8.93 | 20.90 | 56.00 | 21.65 | 12.03 | 24.74 |
| CoworkBench | 83.00 | 25.00 | 78 | 73.79 | 77.82 | 64.11 | 59.27 |
| SkillsBench (no skills) | 33.08 | — | 29.89 | 45.47 | 45.47 | 42.06 | 37.00 |
| SkillsBench (skills) | 58.24 | — | 59.81 | 61.54 | — | 42.06 | 46.90 |
| HealthAdminBench (success rate / mean) | 40.74 / 88.27 | — | 25.92 / 78.36 | 45.93 / 88.37 | — | 41.48 / 86.12 | — |

#### Instant

##### Code

| Benchmark | GigaChat 4 708B | GigaChat 3.1 Ultra 702B | GigaChat 3.5 Ultra 432B | DeepSeek V4 Flash 0731 | Qwen 3.5 397B |
| --- | ---: | ---: | ---: | ---: | ---: |
| LiveCodeBench v6 | 78.77 | 57.00 | 56.20 | 71.18 | 61.52 |
| SWE-bench Verified (7-hour timeout) | 80.82 | 9.00 | 39.70 | 78.80 | 68.60 |
| SWE-bench Multilingual (7-hour timeout) | 70.67 | 3.33 | 21.60 | 72.33 | 65.00 |
| DeepSWE (7-hour timeout) | 19.47 | 0.00 | 0.00 | 28.32 | 0.88 |
| NL2Repo (Full score) (7-hour timeout) | 37.56 | 7.90 | — | 30.37 | 30.28 |
| Terminal-Bench 2.0 (7-hour timeout) | 60.47 | 9.00 | 12.94 | — | 49.44 |
| Terminal-Bench 2.1 (7-hour timeout) | 66.29 | 6.74 | — | 70.77 | 56.18 |

##### General

| Benchmark | GigaChat 4 708B | GigaChat 3.1 Ultra 702B | GigaChat 3.5 Ultra 432B | DeepSeek V4 Flash 0731 | Qwen 3.5 397B |
| --- | ---: | ---: | ---: | ---: | ---: |
| IFBench (prompt-level loose) | 50.33 | 29.00 | 43.66 | 40.00 | 53.00 |
| JSONSchemaBench | 88.31 | 93.96 | 83.94 | 93.70 | 93.45 |
| StructEval Text | 89.12 | 84.98 | 74.35 | 85.64 | 83.62 |
| POLLUX | 79.65 | 63.28 | 74.25 | 72.22 | 74.29 |
| GraphWalks | 48.02 | 15.16 | 19.99 | 51.16 | 56.49 |
| Natural Plan | 52.17 | 32.00 | 66.42 | 41.58 | 45.72 |
| NIH Complex | 95.71 | 74.22 | 97.11 | 32.21 | 99.24 |
| AA-LCR | 48.00 | — | 27.60 | — | 60.00 |

##### STEM

| Benchmark | GigaChat 4 708B | GigaChat 3.1 Ultra 702B | GigaChat 3.5 Ultra 432B | DeepSeek V4 Flash 0731 | Qwen 3.5 397B |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPQA Diamond | 70.20 | 56.57 | 61.11 | 74.75 | 83.33 |
| HMMT (pass@8) | 86.67 | 45.00 | 55.00 | 70.00 | 78.33 |
| HMMT (mean@8) | 63.75 | — | — | 44.38 | 63.33 |
| AIME 2025 (pass@8) | 93.33 | 73.33 | 93.30 | 86.67 | 93.33 |
| AIME 2025 (mean@8) | 68.75 | 37.08 | 68.00 | 48.23 | 80.00 |
| AIME 2026 (pass@8) | 96.67 | 66.67 | 83.30 | 93.33 | 93.333 |
| AIME 2026 (mean@8) | 80.83 | 39.17 | 67.00 | 60.42 | 86.250 |
| IMO-AnswerBench | 60.09 | 28.75 | 32.00 | 41.91 | 72.38 |
| Humanity's Last Exam (text only, without tools) | 8.20 | 4.00 | 3.94 | 10.43 | 19.83 |

##### Agentic

| Benchmark | GigaChat 4 708B | GigaChat 3.1 Ultra 702B | GigaChat 3.5 Ultra 432B | DeepSeek V4 Flash 0731 | Qwen 3.5 397B |
| --- | ---: | ---: | ---: | ---: | ---: |
| BFCL v4 | 63.62 | 77.00 | 51.57 | 65.81 | — |
| τ²-Bench (mean@3) | 66.75 | 33.00 | 68.71 | 77.13 | 78.73 |
| τ³-Bench Banking (mean@3) | 15.00 | 5.50 | 5.15 | — | 12.71 |
| CoworkBench (Pi) | 75.00 | — | 19.76 | 54.84 | 63.31 |
| SkillsBench (without skills) | 33.93 | 0.51 | — | 27.87 | — |
| SkillsBench (with skills) | 54.24 | 1.23 | — | 54.32 | — |
| HealthAdminBench (success rate / mean) | 40.74 / 86.99 | — | — | 23.73 / 70.14 | 33.33 / — |
| JobBench | 33.59 | — | — | 29.64 | 16.85 |

#### Evaluation Notes

Agentic benchmarks were evaluated using Harbor-based runners (SWE-bench, Terminal-Bench, NL2Repo, SkillsBench, HealthAdminBench, CoworkBench, and JobBench), except DeepSWE, which used Pier. Runs with a seven-hour timeout are marked in the tables.

NL2Repo reports cheating-adjusted scores, counting cheating attempts as zero. StructEval uses the text subset with rule-based scoring, while HLE includes only text-based questions without tools. HealthAdminBench reports success rate / mean.

---

## GigaChat 4 37B-A4B

GigaChat 4 37B-A4B brings the same hybrid architecture to a smaller model, with 37B total parameters and 4B active parameters per token. It is designed for efficient deployment while supporting multilingual conversation, coding, reasoning, and agentic tasks.

Key highlights:

- **37B total / 4B active parameters per token**
- **32 decoder layers**, including 24 GatedDeltaNet and 8 DSA layers
- **1M-token context window** in the final base and instruction-tuned models
- **Instant and reasoning modes** available within a single instruction-tuned checkpoint
- **FP8, BF16, and GGUF** instruction-tuned weights, plus base and intermediate pretraining checkpoints

### Intermediate checkpoints

The 37B checkpoints repository includes the following training revisions:

| Stage | Step | Branch | Configured context |
| --- | --- | --- | --- |
| Pretraining | 200k | [pretrain-200k-step](https://huggingface.co/ai-sage/GigaChat4-37B-A4B-checkpoints/tree/pretrain-200k-step) | 4,096 |
| Pretraining | 300k | [pretrain-300k-step](https://huggingface.co/ai-sage/GigaChat4-37B-A4B-checkpoints/tree/pretrain-300k-step) | 4,096 |
| Pretraining | 400k | [pretrain-400k-step](https://huggingface.co/ai-sage/GigaChat4-37B-A4B-checkpoints/tree/pretrain-400k-step) | 4,096 |
| Midtraining | 510k | [midtrain-510k-step](https://huggingface.co/ai-sage/GigaChat4-37B-A4B-checkpoints/tree/midtrain-510k-step) | 32,768 |
| Start of context extension | 518k | [lc-start-518k-step](https://huggingface.co/ai-sage/GigaChat4-37B-A4B-checkpoints/tree/lc-start-518k-step) | 32,768 |

The 1M-token context is available in the final [37B base model](https://huggingface.co/ai-sage/GigaChat4-37B-A4B-base) and the final instruction-tuned models. The intermediate revisions listed above have 4K or 32K configured context lengths.

### Benchmarks

The following tables summarize the performance of the final instruction-tuned GigaChat 4 37B-A4B model. Scores are reported as percentages unless otherwise specified. A dash indicates an unreported result.

#### Reasoning

##### Code

| Benchmark | GigaChat 4 37B | GigaChat 3.5 37B | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B | GigaChat 3.5 Ultra 432B |
| --- | ---: | ---: | ---: | ---: | ---: |
| LiveCodeBench v6 | 85.00 | 64.00 | 69.00 | 65.00 | 85.40 |
| SWE-bench Verified | 70.80 | 32.40 | 60.60 | 67.80 | 64.70 |
| SWE-bench Multilingual | 61.67 | 26.00 | 57.30 | 60.70 | — |
| NL2Repo (mean score / pass rate) | 14.89 / 25.03 | 3.20 / 3.40 | 11.40 / 5.70 | 30.30 / 17.70 | 5.00 / — |
| Terminal-Bench 2.0 | 38.20 | 22.50 | 33.70 | 48.30 | 30.30 |
| Terminal-Bench 2.1 | 39.33 | 22.50 | 34.80 | 49.40 | — |

##### General

| Benchmark | GigaChat 4 37B | GigaChat 3.5 37B | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B | GigaChat 3.5 Ultra 432B |
| --- | ---: | ---: | ---: | ---: | ---: |
| IFBench (prompt-level loose) | 75.00 | 61.00 | 64.30 | 58.00 | 77.00 |
| JSONSchemaBench | 96.00 | 87.00 | 94.20 | 42.00 | 93.01 |
| StructEval Text | 86.00 | 75.00 | 78.90 | 85.60 | 85.00 |
| POLLUX | — | 73.00 | 68.60 | 71.00 | 75.50 |
| GraphWalks | 63.00 | 13.00 | 60.50 | 46.00 | 38.00 |
| Natural Plan | 92.00 | 51.00 | 75.20 | 76.00 | 80.19 |
| NIH Complex | 99.00 | 94.00 | 96.90 | 67.00 | 99.16 |
| AA-LCR | 54.00 | 18.70 | 66.00 | 60.70 | 25.00 |

##### STEM

| Benchmark | GigaChat 4 37B | GigaChat 3.5 37B | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B | GigaChat 3.5 Ultra 432B |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPQA Diamond | 70.00 | 66.00 | 85.00 | 80.00 | 82.30 |
| HMMT (pass@8) | 95.00 | 86.70 | 87.00 | 83.00 | 93.00 |
| HMMT (mean@8) | 87.00 | 61.50 | 74.00 | 70.00 | — |
| AIME 2025 (pass@8) | 93.00 | 80.00 | 93.00 | 93.00 | 93.00 |
| AIME 2025 (mean@8) | 88.00 | 73.00 | 88.00 | 92.00 | 89.00 |
| AIME 2026 (pass@8) | 93.00 | 93.00 | 97.00 | 97.00 | 97.00 |
| AIME 2026 (mean@8) | 92.00 | 78.00 | 88.00 | 87.00 | 91.00 |
| IMO-AnswerBench | 76.00 | 60.00 | 74.00 | 76.00 | 73.00 |
| HLE (text only) | 9.00 | 6.00 | 19.00 | 20.00 | 13.00 |

##### Agentic

| Benchmark | GigaChat 4 37B | GigaChat 3.5 37B | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B | GigaChat 3.5 Ultra 432B |
| --- | ---: | ---: | ---: | ---: | ---: |
| BFCL (overall accuracy) | 72.00 | 63.00 | 69.00 | 67.00 | 58.59 |
| τ³-Bench (mean@3) | 57.00 | 41.20 | 53.60 | 63.30 | 47.80 |
| τ²-Bench (mean@3) | 73.00 | 53.40 | 69.40 | 80.00 | 61.58 |
| τ³-Bench Banking (mean@3) | 9.00 | 4.80 | 6.20 | 8.60 | 8.93 |
| CoworkBench (Pi) | 59.48 | 5.80 | 39.00 | 57.00 | 25.00 |
| Toolathlon | 39.00 | 18.90 | 40.00 | 42.00 | 1.41 |
| SkillsBench (without skills) | 8.62 | 1.20 | 9.00 | 14.00 | — |
| SkillsBench (with skills) | 23.80 | 3.70 | 25.00 | 35.00 | — |
| JobBench | 23.28 | 2.50 | 9.10 | 17.10 | — |

#### Instant

##### Code

| Benchmark | GigaChat 4 37B | GigaChat 3.5 37B | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B | GigaChat 3.5 Ultra 432B |
| --- | ---: | ---: | ---: | ---: | ---: |
| LiveCodeBench v6 | 52.00 | 49.00 | 57.63 | 46.45 | 56.20 |
| SWE-bench Verified | 70.80 | 34.00 | 57.80 | 63.40 | 39.70 |
| SWE-bench Multilingual | 61.67 | 7.00 | 54.30 | 59.70 | 21.60 |
| Terminal-Bench 2.0 | 38.20 | 5.60<sup>*</sup> | 36.00 | 44.90<sup>*</sup> | 12.94 |
| Terminal-Bench 2.1 | 39.33 | 7.90 | 46.10 | 46.10<sup>*</sup> | — |

##### General

| Benchmark | GigaChat 4 37B | GigaChat 3.5 37B | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B | GigaChat 3.5 Ultra 432B |
| --- | ---: | ---: | ---: | ---: | ---: |
| IFBench (prompt-level loose) | 48.00 | 35.00 | 41.30 | 37.30 | 43.66 |
| JSONSchemaBench | 86.00 | 85.00 | 91.00 | 92.10 | 83.94 |
| StructEval Text | 87.00 | 72.00 | 83.80 | 82.40 | 74.35 |
| POLLUX | — | 70.00 | 65.10 | — | 74.25 |
| GraphWalks | 26.00 | 7.00 | 56.20 | 55.60 | 19.99 |
| Natural Plan | 45.00 | 21.00 | 44.10 | 44.60 | 66.42 |
| NIH Complex | 92.00 | 82.00 | 96.60 | 99.99 | 97.11 |
| AA-LCR | 54.00 | 10.70 | 52.70 | 52.30 | 27.60 |

##### STEM

| Benchmark | GigaChat 4 37B | GigaChat 3.5 37B | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B | GigaChat 3.5 Ultra 432B |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPQA Diamond | 55.00 | 58.00 | 78.30 | 81.80 | 61.11 |
| HMMT (pass@8) | 67.00 | 73.30 | 71.70 | 80.00 | 55.00 |
| HMMT (mean@8) | — | 46.90 | 49.80 | 56.70 | — |
| AIME 2025 (pass@8) | 73.00 | 67.00 | 93.30 | 90.00 | 93.00 |
| AIME 2025 (mean@8) | 39.00 | 50.00 | 76.30 | 72.50 | 68.00 |
| AIME 2026 (pass@8) | 80.00 | 77.00 | 93.30 | 93.30 | 83.00 |
| AIME 2026 (mean@8) | 57.00 | 55.00 | 71.30 | 76.30 | 67.00 |
| IMO-AnswerBench | 35.00 | 44.00 | 59.20 | 56.00 | 32.00 |
| HLE (text only) | 4.00 | 4.00 | 12.20 | 12.90 | 4.00 |

##### Agentic

| Benchmark | GigaChat 4 37B | GigaChat 3.5 37B | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B | GigaChat 3.5 Ultra 432B |
| --- | ---: | ---: | ---: | ---: | ---: |
| BFCL (overall accuracy) | 49.00 | 37.00 | 57.80 | 55.00 | 51.57 |
| τ³-Bench (mean@3) | 47.30 | 40.80 | 60.20 | 59.40 | 50.04 |
| τ²-Bench (mean@3) | 60.00 | 54.00 | 78.50 | 78.00 | 68.71 |
| τ³-Bench Banking (mean@3) | 10.00 | 2.10<sup>*</sup> | 5.50 | 3.80 | 5.15 |

#### Evaluation Notes

IFBench uses prompt-level loose scoring. NL2Repo entries report mean score / pass rate, in that order. HLE uses the text subset.

---

## Training and alignment

### Pretraining stability

Several techniques were introduced to improve training stability at scale:

- **Progressive residual warm-up (ProRes)** gradually increases residual branch contributions during early training, with deeper layers warming up more slowly to prevent activation growth.

- **Projection-specific z-loss and logit softcapping** control activation outliers and extreme output logits.

- **Dynamic learning-rate rescaling (DLR)** adjusts learning rates for individual MoE blocks when persistent expert-load imbalance is detected, without affecting the rest of the model.

The **37B model** additionally uses **Muon with μP-based scaling** throughout pretraining and midtraining. We derived scaling rules for Muon's weight updates that account for GatedDeltaNet and gated normalization, allowing hyperparameters tuned on a 4B model to transfer to 37B. During midtraining, we increased the number of Newton–Schulz iterations per optimizer step.

### Alignment

Following supervised fine-tuning, expert models were trained using reinforcement learning on mathematics, coding, tool use, and multi-turn agent workflows. Their capabilities were consolidated into a single model through **multi-teacher on-policy distillation**, where the student generates its own responses and receives task-specific feedback from the corresponding teachers.

Muon was used during this distillation stage for the 708B model. The final instruction-tuned models support both instant and reasoning modes in the same checkpoint.

For details on the 708B model's architecture, training data, context extension, and alignment, see the [GigaChat 4 technical report](https://github.com/salute-developers/gigachat4/blob/main/GigaChat_4.pdf). A dedicated technical report for GigaChat 4 37B will be published later.

---

## Inference and deployment

### SGLang

#### SGLang installation

GigaChat 4 is natively supported in [SGLang](https://github.com/sgl-project/sglang), including recurrent MTP and DFlash2 speculative decoding, per-request instant/reasoning mode selection, and tool calling.

The deployment recipes below were tested on SGLang commit [`ff14289793`](https://github.com/sgl-project/sglang/commit/ff142897935ec6ede004dea8439d6111110bd446). The FP8 deployment examples below enable recurrent MTP speculative decoding using the built-in draft head. For DFlash2 deployment and alternative serving configurations, see the SGLang cookbook.

```bash
python3 -m pip install \
  "sglang @ git+https://github.com/sgl-project/sglang.git@ff142897935ec6ede004dea8439d6111110bd446#subdirectory=python"
```

#### GigaChat 4 708B-A41B: FP8

The FP8 model can be deployed on two nodes with eight H100 80GB GPUs each (TP16). Download the same model revision on both nodes and run the following command with `HEAD_IP` set to the head node's IP address and `NODE_RANK` set to `0` or `1` on the respective nodes.

```bash
sglang serve \
  --model-path ai-sage/GigaChat4-708B-A41B \
  --tp 16 \
  --nnodes 2 \
  --node-rank "$NODE_RANK" \
  --dist-init-addr "${HEAD_IP}:20000" \
  --mem-fraction-static 0.8 \
  --max-running-requests 64 \
  --cuda-graph-max-bs-decode 64 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --host 0.0.0.0 \
  --port 30000
```

#### GigaChat 4 37B-A4B: FP8

The FP8 model can be deployed on a single H100 80GB GPU (TP1). The model weights require approximately 40 GB of GPU memory, with additional memory needed for attention caches, recurrent states, and activations.

```bash
sglang serve \
  --model-path ai-sage/GigaChat4-37B-A4B \
  --tp 1 \
  --mem-fraction-static 0.8 \
  --max-running-requests 64 \
  --cuda-graph-max-bs-decode 64 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --host 0.0.0.0 \
  --port 30000
```

#### BF16, base, and intermediate checkpoints

The BF16 model cards provide separate deployment examples:

| Model | Example GPU configuration | Tensor parallelism |
| --- | --- | --- |
| [708B BF16](https://huggingface.co/ai-sage/GigaChat4-708B-A41B-bf16) | Four nodes, eight H100 80GB GPUs per node | TP32 |
| [37B BF16](https://huggingface.co/ai-sage/GigaChat4-37B-A4B-bf16) | One node, two H100 80GB GPUs | TP2 |

The 708B BF16 weights require approximately 1.4 TB of GPU memory; the 37B BF16 weights require approximately 76.5 GB. Allow additional memory for attention caches, recurrent states, and activations.

Use the base-model cards for raw text completion through `/v1/completions`. For intermediate checkpoints, select the branch with `--revision`. The model cards linked above contain the commands and settings for each variant.

#### Instant and reasoning modes

Both modes are available through the same OpenAI-compatible API. Set `chat_template_kwargs.reasoning` to `false` for instant mode or `true` for reasoning mode.

The request below targets the 37B FP8 server. For another model, replace the `model` value with the ID served by your deployment.

```bash
curl http://localhost:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ai-sage/GigaChat4-37B-A4B",
    "messages": [
      {"role": "user", "content": "What is 17 times 24?"}
    ],
    "temperature": 1.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "max_tokens": 4096,
    "chat_template_kwargs": {"reasoning": false}
  }'
```

The chat template enables reasoning by default when the argument is omitted. With the reasoning parser enabled, thinking is returned in `message.reasoning_content` and the final answer in `message.content`.

For DFlash2, tool calling, multi-turn conversations, streaming, alternative hardware configurations, and performance benchmarks, see the [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/GigaChat/GigaChat4).

### vLLM

GigaChat 4 supports inference with [vLLM](https://github.com/vllm-project/vllm), including recurrent MTP and DFlash2 speculative decoding, per-request instant/reasoning mode selection, and tool calling.

The examples use vLLM commit [`c0c9083`](https://github.com/vllm-project/vllm/commit/c0c90837262bf3059b3d7b20bf120ffbe197a785) from the [GigaChat 4 integration PR](https://github.com/vllm-project/vllm/pull/69762).

The FP8 deployment examples below enable recurrent MTP speculative decoding using the built-in draft head.

#### vLLM installation

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git fetch origin c0c90837262bf3059b3d7b20bf120ffbe197a785
git checkout --detach c0c90837262bf3059b3d7b20bf120ffbe197a785

python3 -m pip install --upgrade pip
VLLM_USE_PRECOMPILED=1 python3 -m pip install -e .
```

#### GigaChat 4 708B-A41B: FP8

The example below uses two nodes with eight H100 80GB GPUs each (TP16) and recurrent MTP speculative decoding enabled.

```bash
# Set HEAD_IP to the head node's IP address and NODE_RANK to 0 or 1.
# Run on both nodes; add --headless on the worker node (NODE_RANK=1).

vllm serve ai-sage/GigaChat4-708B-A41B \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank "$NODE_RANK" \
  --master-addr "$HEAD_IP" \
  --master-port 20000 \
  --no-numa-bind \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096 \
  --compilation-config '{"max_cudagraph_capture_size":64}' \
  --tool-call-parser gigachat4 \
  --enable-auto-tool-choice \
  --reasoning-parser gigachat4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --host 0.0.0.0 \
  --port 8000
```

#### GigaChat 4 37B-A4B: FP8

The example below uses a single H100 80GB GPU (TP1) with recurrent MTP speculative decoding enabled.

```bash
vllm serve ai-sage/GigaChat4-37B-A4B \
  --tensor-parallel-size 1 \
  --no-numa-bind \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --compilation-config '{"max_decode_cudagraph_capture_requests": 64}' \
  --tool-call-parser gigachat4 \
  --enable-auto-tool-choice \
  --reasoning-parser gigachat4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --host 0.0.0.0 \
  --port 8000
```

#### BF16 models

The BF16 model cards provide separate vLLM deployment examples:

| Model | Example GPU configuration | Tensor parallelism |
| --- | --- | --- |
| [708B BF16](https://huggingface.co/ai-sage/GigaChat4-708B-A41B-bf16) | Four nodes, eight H100 80GB GPUs per node | TP32 |
| [37B BF16](https://huggingface.co/ai-sage/GigaChat4-37B-A4B-bf16) | One node, two H100 80GB GPUs | TP2 |

For the 708B BF16 deployment, set `NODE_RANK` to `0`, `1`, `2`, or `3` on the respective nodes and add `--headless` on workers. Use the same `HEAD_IP` and master port on all nodes.

#### Instant and reasoning modes

The request below targets the 37B FP8 server on port `8000`. For another model, replace the `model` value with the ID served by your deployment.

Both modes are available through the same OpenAI-compatible API. Set `chat_template_kwargs.reasoning` to `false` for instant mode or `true` for reasoning mode.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ai-sage/GigaChat4-37B-A4B",
    "messages": [
      {"role": "user", "content": "Prove that the sum of the first n odd positive integers is n**2. Write a proof in the style of Eminem’s lyrics."}
    ],
    "max_tokens": 8192,
    "temperature": 1.0,
    "top_p": 1.0,
    "chat_template_kwargs": {"reasoning": true}
  }'
```

The chat template enables reasoning by default when the argument is omitted. With the reasoning parser enabled, the response separates thinking into `message.reasoning` and the final answer into `message.content`. The `max_tokens` limit covers both.

#### Tool calling

The following example demonstrates tool calling in reasoning mode using the OpenAI-compatible Chat Completions API.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ai-sage/GigaChat4-37B-A4B",
    "messages": [
      {"role": "user", "content": "What is the weather in London right now?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather in a city",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"}
            },
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "max_tokens": 2000,
    "temperature": 0.6,
    "chat_template_kwargs": {"reasoning": true}
  }'
```

When the model requests a tool, the response contains the function name and arguments in `message.tool_calls`. Execute the function in your application, append the assistant message and a `tool` message with the matching `tool_call_id`, then request the final answer.

### llama.cpp

The GGUF releases support GPU and CPU inference with [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/32767). Each weight format includes a recurrent MTP draft head. The example below uses Q4_K_M weights and enables up to three draft tokens per verification step.

#### llama.cpp installation

Fetch the GigaChat 4 integration PR and build the CUDA server. Run the installation, download, and server commands in the same shell from the `llama.cpp` directory.

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git fetch origin pull/32767/head:pr-32767
git checkout pr-32767

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_USE_PREBUILT_UI=OFF # build for CUDA
cmake --build build --config Release --parallel --target llama-server
```

#### Download GGUF weights

Choose one model below. If authentication is required, run `hf auth login` with an account that has access before downloading.

**37B-A4B:** Q4_K_M is a single 23.3 GB GGUF file.

```bash
python -m pip install -U huggingface_hub

hf download ai-sage/GigaChat4-37B-A4B-GGUF \
  --include "GigaChat4-37B-A4B-Q4_K_M/*.gguf" \
  --local-dir ./gguf

MODEL=./gguf/GigaChat4-37B-A4B-Q4_K_M/GigaChat4-37B-A4B-Q4_K_M.gguf
```

**708B-A41B:** Q4_K_M contains nine shards totaling 429.4 GB. Download all shards to the same directory and pass the first shard to the server.

```bash
python -m pip install -U huggingface_hub

hf download ai-sage/GigaChat4-708B-A41B-GGUF \
  --include "GigaChat4-708B-A41B-Q4_K_M/*.gguf" \
  --local-dir ./gguf

MODEL=./gguf/GigaChat4-708B-A41B-Q4_K_M/GigaChat4-708B-A41B-Q4_K_M-00001-of-00009.gguf
```

#### Run the server

The command below offloads all layers to the available GPUs and starts four slots with 32,768 tokens of total context. Allow GPU memory for the weights, KV cache, and compute buffers.

```bash
./build/bin/llama-server \
  -m "$MODEL" --alias gigachat4 \
  -sm layer -ngl 99 --fit off -fa on \
  -c 32768 -np 4 -ctk q8_0 -ctv q8_0 \
  --jinja --reasoning-format deepseek \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --host 127.0.0.1 --port 8080
```

The OpenAI-compatible Chat Completions endpoint is `http://127.0.0.1:8080/v1/chat/completions`; use `"model": "gigachat4"` in requests. Set `chat_template_kwargs.reasoning` to `false` for instant mode or `true` for reasoning mode. To disable speculative decoding, replace `--spec-type draft-mtp --spec-draft-n-max 3` with `--spec-type none`.

For CPU builds, other GGUF formats, and tool-calling examples, see the [37B GGUF model card](https://huggingface.co/ai-sage/GigaChat4-37B-A4B-GGUF#usage) and [708B GGUF model card](https://huggingface.co/ai-sage/GigaChat4-708B-A41B-GGUF#usage).

---

## Recommended agent harnesses

We recommend Pi for agentic tasks with GigaChat 4. Codex CLI and DeepSeek Harness are alternatives for interactive terminal work. You can use each harness’s standard prompts and tools without fine-tuning the checkpoint for that harness.

We verified interactive shell tool calls on an internal checkpoint with Pi 1.6.0, Codex CLI 0.171.0, and DeepSeek Harness 0.2.2-rc.5. Pi and DeepSeek Harness connected directly through Chat Completions. For Codex, we used a Responses-to-Chat-Completions gateway.

Start SGLang as shown above. The examples use the API base URL `http://127.0.0.1:30000/v1` and the model ID `ai-sage/GigaChat4-708B-A41B`. For the vLLM deployment above, use the base URL `http://127.0.0.1:8000/v1`. For the 37B model, use `ai-sage/GigaChat4-37B-A4B` throughout. For a BF16 deployment, use the corresponding `-bf16` model ID. If you set a custom served name, use the ID returned by `GET /v1/models`.

| Harness | API used by the harness | Required endpoint |
| --- | --- | --- |
| Pi | OpenAI Chat Completions | `POST /v1/chat/completions` |
| Codex CLI | OpenAI Responses | `POST /v1/responses` |
| DeepSeek Harness | OpenAI Chat Completions | `POST /v1/chat/completions` |

### Pi

Add `gigachat` to the `providers` object in `~/.pi/agent/models.json`.

```json
{
  "providers": {
    "gigachat": {
      "baseUrl": "http://127.0.0.1:30000/v1",
      "api": "openai-completions",
      "apiKey": "$GIGACHAT_API_KEY",
      "models": [
        { "id": "ai-sage/GigaChat4-708B-A41B" }
      ]
    }
  }
}
```

Pi reads `GIGACHAT_API_KEY` from the environment. Export your endpoint’s key, or use `export GIGACHAT_API_KEY=local` for an unauthenticated local server. Run `pi` from your project directory and select GigaChat 4 with `/model`.

### Codex CLI

Codex requires a Responses API endpoint that can stream tool calls and accept tool results on subsequent requests. Connect directly if your SGLang deployment supports the tool types sent by your Codex version. If it serves only Chat Completions or rejects those tool types, set `base_url` to the API root of a compatible Responses-to-Chat-Completions gateway.

For Codex CLI 0.134.0 and later, save this profile as `~/.codex/gigachat.config.toml`.

```toml
model = "ai-sage/GigaChat4-708B-A41B"
model_provider = "gigachat"

[model_providers.gigachat]
name = "GigaChat 4"
base_url = "http://127.0.0.1:30000/v1"
wire_api = "responses"
```

For an authenticated endpoint, add `env_key = "GIGACHAT_API_KEY"` under `[model_providers.gigachat]` and export the key. If your gateway rejects the built-in `web_search` tool, add `web_search = "disabled"` at the top level, above `[model_providers.gigachat]`. We used this setting with our gateway.

<details>
<summary>Codex CLI before 0.134.0</summary>

Put both sections in `~/.codex/config.toml`.

```toml
[model_providers.gigachat]
name = "GigaChat 4"
base_url = "http://127.0.0.1:30000/v1"
wire_api = "responses"

[profiles.gigachat]
model = "ai-sage/GigaChat4-708B-A41B"
model_provider = "gigachat"
```

When upgrading, move the profile settings into `~/.codex/gigachat.config.toml`, then remove the old `[profiles.gigachat]` section and any top-level `profile = "gigachat"` setting from the base file.

</details>

Start an interactive session from your project directory.

```bash
codex --profile gigachat
```

### DeepSeek Harness

For the interactive terminal UI, install the tested launcher and plugin versions.

```bash
npm install -g @deepseek-ai/dsh@0.2.2-rc.5
dsh plugin --profile tui add @deepseek-harness-tui/dsh-tui@0.20.2
```

Save the following as `gigachat-dsh.yml`. The interactive TUI and the headless CLI use the same configuration.

```yaml
- id: llm-pi-ai
  config:
    providers:
      gigachat:
        api: openai-completions
        baseURL: http://127.0.0.1:30000/v1
        apiKeyEnv: GIGACHAT_API_KEY
        models:
          - id: ai-sage/GigaChat4-708B-A41B

- id: agent-default-model
  config:
    provider: gigachat
    model: ai-sage/GigaChat4-708B-A41B
```

Export `GIGACHAT_API_KEY` with your endpoint’s key, or set it to `local` for an unauthenticated server. Launch the TUI from the directory containing the configuration.

```bash
dsh --profile tui --patch ./gigachat-dsh.yml
```

### Verify the connection and tool calling

To check that the harness can reach the model, execute a tool, and return its output to the model, ask this question in an interactive session:

> How much RAM is available on this machine right now?

The model chooses how to inspect memory. Look for a shell tool call in the session and check that the answer matches the available RAM in the command output.

You can run the same check from a script with the DeepSeek Harness headless CLI.

```bash
dsh --profile headless --patch ./gigachat-dsh.yml \
  'How much RAM is available on this machine right now?'
```

---

## License

The two model sizes are released under different licenses:

| Model family | License |
| --- | --- |
| GigaChat 4 708B-A41B | [MIT License](https://huggingface.co/ai-sage/GigaChat4-708B-A41B/blob/main/LICENSE.txt) |
| GigaChat 4 37B-A4B | [Noncommercial License for individuals 1.0](https://huggingface.co/ai-sage/GigaChat4-37B-A4B/blob/main/LICENSE.txt) |

The model repositories contain the applicable license for each checkpoint and weight format.
