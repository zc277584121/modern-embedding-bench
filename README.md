# mm-embedding-bench

A framework for evaluating and comparing multimodal embedding models across multiple providers and tasks.

## Features

- **10+ embedding providers** — API services (OpenAI, Gemini, Voyage, Cohere, Jina, DashScope, Volcengine/ARK) and local models (SentenceTransformers, Transformers, Ollama)
- **4 evaluation tasks** — MRL stress test, cross-modal retrieval, crosslingual retrieval, needle-in-a-haystack
- **Disk-based embedding cache** — avoids redundant API calls and GPU computation
- **Incremental result saving** — results saved after each model-task combination

## Supported Providers

| Provider | Example Model | Modalities |
|----------|--------------|------------|
| **OpenAI** | text-embedding-3-large | Text |
| **Gemini** | gemini-embedding-2-preview | Text, Image, Video, Audio, PDF |
| **Voyage** | voyage-multimodal-3.5 | Text, Image |
| **Cohere** | embed-v4.0 | Text, Image, Document |
| **Jina** | jina-embeddings-v4, jina-clip-v2 | Text, Image, Document |
| **DashScope** | text-embedding-v3, multimodal-embedding-v1 | Text, Image |
| **Volcengine/ARK** | doubao-embedding | Text |
| **SentenceTransformers** | BAAI/bge-m3, clip-ViT-B-32 | Text (+ Image for CLIP) |
| **Transformers** | Qwen3-VL-Embedding-2B, SigLIP2 | Text, Image |
| **Ollama** | nomic-embed-text, bge-m3 | Text |

## Evaluation Tasks

| Task | Description | Modalities |
|------|-------------|------------|
| **MRL Stress** | Matryoshka dimension reduction quality (Spearman ρ) | Text |
| **Cross-Modal Retrieval** | Bidirectional text ↔ image retrieval with hard negatives | Text + Image |
| **Crosslingual Retrieval** | Chinese ↔ English parallel sentence retrieval | Text |
| **Needle-in-a-Haystack** | Specific fact retrieval in long documents (1K–32K chars) | Text |

## Installation

```bash
# Clone and install with uv
git clone https://github.com/your-org/mm-embedding-bench.git
cd mm-embedding-bench
uv sync
```

## Quick Start

```python
from mm_embed.providers import get_provider
from mm_embed.tasks import get_task

# Initialize a provider
provider = get_provider("openai", model="text-embedding-3-large")

# Run a task
task = get_task("mrl_stress")
result = task.run(provider)
print(result.metrics)
```

Or use the evaluation scripts:

```bash
# Run all evaluations
uv run python scripts/run_rerun_all.py

# Run specific evaluations
uv run python scripts/run_crosslingual_eval.py
uv run python scripts/run_crossmodal_hard.py
```

## Environment Variables

```bash
# API keys (set whichever providers you need)
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
export VOYAGE_API_KEY="..."
export COHERE_API_KEY="..."
export JINA_API_KEY="..."
export DASHSCOPE_API_KEY="..."
export ARK_API_KEY="..."

# Optional: GPU device for local models (default: cuda:0)
export CUDA_DEVICE="cuda:0"

# Optional: local model paths (default: HuggingFace model names)
export QWEN_VL_MODEL_PATH="Qwen/Qwen3-VL-Embedding-2B"
export SIGLIP_MODEL_PATH="google/siglip2-so400m-patch14-384"
```

## Project Structure

```
mm-embedding-bench/
├── pyproject.toml
├── src/mm_embed/
│   ├── cache.py                          # Disk-based embedding cache
│   ├── cli.py                            # CLI entry point
│   ├── providers/
│   │   ├── base.py                       # EmbeddingProvider ABC
│   │   ├── registry.py                   # Lazy provider registry
│   │   ├── openai_provider.py            # OpenAI
│   │   ├── gemini_provider.py            # Google Gemini
│   │   ├── voyage_provider.py            # Voyage AI
│   │   ├── cohere_provider.py            # Cohere
│   │   ├── jina_provider.py              # Jina AI
│   │   ├── dashscope_provider.py         # Alibaba DashScope
│   │   ├── ark_provider.py               # Volcengine/ByteDance ARK
│   │   ├── ollama_provider.py            # Ollama (local)
│   │   ├── sentence_transformers_provider.py  # SentenceTransformers (local GPU)
│   │   └── transformers_provider.py      # HuggingFace Transformers (local GPU)
│   ├── tasks/
│   │   ├── base.py                       # EvalTask ABC + EvalResult
│   │   ├── registry.py                   # Lazy task registry
│   │   ├── mrl_stress.py                 # MRL dimension reduction test
│   │   ├── cross_modal_retrieval.py      # Text ↔ Image retrieval
│   │   ├── crosslingual_retrieval.py     # Chinese ↔ English retrieval
│   │   └── needle_in_haystack.py         # Long-doc needle search
│   ├── data/
│   │   ├── mock.py                       # Mock data generators
│   │   └── real_data.py                  # Real dataset loaders
│   └── utils/
│       └── metrics.py                    # Cosine similarity, Recall@K, etc.
├── scripts/                              # Evaluation runner scripts
├── data/                                 # Datasets and embedding cache (gitignored)
└── results/                              # Evaluation results (gitignored)
```

## License

MIT
