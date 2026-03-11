# mm-embedding-bench

Multimodal Embedding Model Benchmark & Evaluation Framework

## Overview

A framework for evaluating and comparing multimodal embedding models across multiple providers and tasks.

## Supported Providers

| Provider | Model | Modalities | China Mainland |
|----------|-------|------------|:--------------:|
| **DashScope** | Qwen3-VL-Embedding-8B | Text, Image, Video | Direct |
| **Volcengine** | Seed-1.6-Embedding | Text, Image, Video | Direct |
| **Gemini** | gemini-embedding-exp-03-07 | Text, Image, Video, Audio, PDF | VPN |
| **Voyage** | voyage-multimodal-3.5 | Text, Image, Video, Document | VPN |
| **Cohere** | embed-v4.0 | Text, Image, Document | VPN |
| **OpenAI** | text-embedding-3-large | Text only | VPN/Azure |
| **Jina** | jina-embeddings-v4 | Text, Image, Document | VPN |

## Evaluation Tasks

| Task | Description | Modalities |
|------|-------------|------------|
| **MRL Stress** | Matryoshka dimension reduction quality test | Text |
| **Cross-Modal Retrieval** | Bidirectional text↔image retrieval | Text + Image |
| **Needle-in-Haystack** | Long document specific fact retrieval | Text |
| **Autonomous Driving** | Domain-specific scene retrieval (CoVLA-style) | Text + Image |
| **Chinese Multimodal** | Chinese text + cross-lingual alignment | Text + Image |

## Setup

```bash
# Install with uv
uv sync

# Install with specific provider dependencies
uv sync --extra dashscope
uv sync --extra all
```

## Usage

```bash
# List available providers and tasks
mm-bench list-providers
mm-bench list-tasks

# Check provider connectivity
mm-bench check dashscope

# Run evaluation
mm-bench run --provider dashscope --task mrl_stress cross_modal_retrieval
mm-bench run --config configs/default.yaml --output results.json

# Or use the script directly
python scripts/run_eval.py --provider dashscope --task mrl_stress
```

## Environment Variables

```bash
# Required API keys (set in ~/.bashrc or .env)
export DASHSCOPE_API_KEY="..."      # Alibaba DashScope
export ARK_API_KEY="..."            # ByteDance Volcengine
export GEMINI_API_KEY="..."         # Google Gemini
export VOYAGE_API_KEY="..."         # Voyage AI
export COHERE_API_KEY="..."         # Cohere
export OPENAI_API_KEY="..."         # OpenAI (baseline)
export JINA_API_KEY="..."           # Jina AI
```

## Project Structure

```
mm-embedding-bench/
├── pyproject.toml                  # Project config (uv/hatch)
├── configs/
│   └── default.yaml                # Default eval configuration
├── scripts/
│   └── run_eval.py                 # Quick evaluation script
├── src/mm_embed/
│   ├── cli.py                      # CLI entry point
│   ├── providers/
│   │   ├── base.py                 # EmbeddingProvider ABC
│   │   ├── registry.py             # Lazy provider registry
│   │   ├── dashscope_provider.py   # Qwen3 (Alibaba)
│   │   ├── volcengine_provider.py  # Seed-1.6 (ByteDance)
│   │   ├── gemini_provider.py      # Gemini (Google)
│   │   ├── voyage_provider.py      # Voyage Multimodal
│   │   ├── cohere_provider.py      # Cohere Embed v4
│   │   ├── openai_provider.py      # OpenAI (baseline)
│   │   └── jina_provider.py        # Jina v4
│   ├── tasks/
│   │   ├── base.py                 # EvalTask ABC
│   │   ├── registry.py             # Lazy task registry
│   │   ├── mrl_stress.py           # MRL dimension reduction test
│   │   ├── cross_modal_retrieval.py# Text↔Image retrieval
│   │   ├── needle_in_haystack.py   # Long-doc needle search
│   │   ├── autonomous_driving.py   # Driving scene retrieval
│   │   └── chinese_multimodal.py   # Chinese + cross-lingual
│   ├── data/
│   │   └── mock.py                 # Mock data generators
│   └── utils/
│       └── metrics.py              # Eval metrics (Recall@K, MRR, etc.)
└── tests/
```
