# Modern IR Bench

Modern IR Bench is a code-native benchmark for modern information retrieval solutions. A participant can be a model, an algorithm, a multi-stage retrieval pipeline, a RAG system, or an agent memory system.

The project is intentionally Python-only: executable benchmark modules are the source of truth, and every published score links back to an immutable Git commit and the exact code that produced it.

## Framework preview

The first vertical slice contains deterministic synthetic data for two tasks:

- Agent Memory Retrieval
- Code Localization

It compares three mock solutions—BM25, character n-grams, and a composed hybrid—and generates the data used by the public read-only Hugging Face Space. These numbers validate the framework and UI; they are not scientific benchmark results.

```bash
uv sync
uv run python -m benchmarks.mock_showcase
uv run pytest
```

Run the Space locally:

```bash
uv sync --extra space
uv run --extra space python space/app.py
```

## Core model

```text
Task
├── declares its Hugging Face Dataset contract
├── owns metrics and evaluation semantics
└── decides how to call a Solution

Solution
└── model, algorithm, retriever, pipeline, or agent being evaluated

Observation
└── sample-level evidence produced by a Task

Metric
└── scores task-specific Observations
```

Datasets use Hugging Face `Dataset` and `IterableDataset` directly. Adapters are ordinary Python functions, and ingestion/search phases remain explicit inside each task and solution.

## Add an experiment

Create an executable module under `benchmarks/`, construct datasets, tasks, metrics, and solutions, then run it directly:

```bash
uv run python -m benchmarks.your_experiment
```

No YAML registry or parallel configuration language is required.

## Publishing model

The Space is a read-only view of maintainer-approved results. Contributions arrive through GitHub pull requests; there is no public model upload or shared evaluation runtime. Dataset releases are immutable, and new data is published as a new version.

The full refactor design is maintained locally in `.local.PLAN.md`.

## Historical implementation

The previous `mm_embed` implementation and historical benchmark assets remain in the repository temporarily for reference. They are not part of the new public package or API.
