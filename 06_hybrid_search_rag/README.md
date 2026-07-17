# Hybrid Search RAG

> Production RAG that actually works.

Combines BM25 + dense + ColBERT late fusion with query rewriting, cross-encoder reranking, and reciprocal rank fusion. Achieves 23% nDCG@10 gain over dense-only approaches.

**Part of [AEGIS](https://github.com/KnigguKniggu-droid/AEGIS)** — Adaptive AI Governance Infrastructure for Cyber-Physical Systems. This subsystem maps to **L3: Multi-Modal Signal Processing** (Sensor Fusion + FFT — multi-modal signal processing combining sparse (BM25) and dense (embedding) representations with late fusion.).

---

## Architecture Position

```
AEGIS Layer: L3: Multi-Modal Signal Processing
ECE Mapping: Sensor Fusion + FFT — multi-modal signal processing combining sparse (BM25) and dense (embedding) representations with late fusion.
```

This module is one of 15 subsystems in the AEGIS platform. See the [unified architecture](https://github.com/KnigguKniggu-droid/AEGIS#readme) for how all components interconnect.

---

## Features

- Query rewrite to BM25 (Postgres) + Dense (Qdrant) + ColBERT (late fusion)
- Cross-encoder reranker (bge-reranker-large) on top-50
- Reciprocal rank fusion with learned weights per corpus
- BEIR + custom benchmark evaluation; 23% nDCG@10 gain
- Streamlit UI: ingest, query, inspect retrieval pipeline per step

---

## Tech Stack

`Python` | `Qdrant` | `PostgreSQL/pgvector` | `Sentence-Transformers` | `FlagEmbedding` | `Streamlit`

---

## Quick Start

```bash
git clone https://github.com/KnigguKniggu-droid/06-hybrid-search-rag.git
cd 06-hybrid-search-rag
pip install -e .
```

Run tests:

```bash
pytest tests/ -v
```

---

## Project Structure

```
06_hybrid_search_rag/
  src/                  # Core application code
  tests/                # 7 passing tests
  .github/              # CI/CD workflows
  Dockerfile            # Container build
  pyproject.toml        # Package configuration
```

---

## Running in Docker

```bash
docker build -t 06_hybrid_search_rag .
docker run -p 8000:8000 06_hybrid_search_rag
```

---

## ECE Design Principles

This subsystem is modeled after classical electrical and computer engineering concepts:

> **Sensor Fusion + FFT — multi-modal signal processing combining sparse (BM25) and dense (embedding) representations with late fusion.**

The AEGIS platform applies safety-critical engineering principles from integrated circuit design to LLM deployment, ensuring production reliability in autonomous vehicles, power grids, and medical devices.

---

## Related Projects

All 15 AEGIS subsystems:

| # | Project | Layer | ECE Mapping |
|---|---------|-------|-------------|
| 01 | [Model Regression Detection](https://github.com/KnigguKniggu-droid/AEGIS) | L5 | SPC |
| 02 | [LLM Cost Autopilot](https://github.com/KnigguKniggu-droid/AEGIS) | L1 | DVFS |
| 03 | [Failure Forensics](https://github.com/KnigguKniggu-droid/AEGIS) | L4 | BIST+ATPG |
| 04 | [Self-Healing Docs](https://github.com/KnigguKniggu-droid/AEGIS) | L6 | ECO |
| 05 | [Output Arbitration](https://github.com/KnigguKniggu-droid/AEGIS) | L4 | TMR |
| 06 | [Hybrid Search RAG](https://github.com/KnigguKniggu-droid/AEGIS) | L3 | Sensor Fusion |
| 07 | [Semantic Cache](https://github.com/KnigguKniggu-droid/AEGIS) | L2 | CAM |
| 08 | [SQL Guardrails](https://github.com/KnigguKniggu-droid/AEGIS) | L4 | MPU/MMU |
| 09 | [A/B Testing](https://github.com/KnigguKniggu-droid/AEGIS) | L5 | SPRT |
| 10 | [LoRA Pipeline](https://github.com/KnigguKniggu-droid/AEGIS) | L1 | SVD |
| 11 | [API Gateway](https://github.com/KnigguKniggu-droid/AEGIS) | L2 | Token Bucket |
| 12 | [Feature Flags](https://github.com/KnigguKniggu-droid/AEGIS) | L5 | FPGA Reconfig |
| 13 | [Dataset Generator](https://github.com/KnigguKniggu-droid/AEGIS) | L3 | Signal Conditioning |
| 14 | [Workflow Orchestrator](https://github.com/KnigguKniggu-droid/AEGIS) | L6 | FSM Sequencer |
| 15 | [LLM Observability](https://github.com/KnigguKniggu-droid/AEGIS) | L7 | Oscilloscope+SA |

---

## License

MIT
