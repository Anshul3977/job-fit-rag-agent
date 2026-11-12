# Eval results

Synthetic labeled set. These numbers were produced by `python eval/run_eval.py` and are not targets.

- Pairs: 20 scored 20, failures 0
- Provider: `deterministic` / `deterministic-v1`
- Embeddings: `fastembed` / `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: `chroma`
- Machine: Linux-6.12.113+-x86_64-with-glibc2.39; processor `x86_64`; CPUs 4; Python 3.11.18
- Wall clock: 2.6 s

## Metrics

| Metric | Value |
| --- | --- |
| Verdict accuracy | 100.0% |
| Adjacent agreement (off-by-one = 0.5) | 100.0% |
| Majority-class baseline (skip) | 60.0% |
| Required-skill hit rate | 100.0% |
| Unsupported-skill rejection rate | 100.0% |
| Citation validity | 100.0% |
| Lexical recall@5 | 100.0% |
| Paraphrase-probe recall@5 | 100.0% |
| Fit latency p50 | 25.0 ms |
| Fit latency p95 | 30.1 ms |
| Retrieve p50 | 20.0 ms |
| Generate p50 | 1.0 ms |
| Mean prompt tokens | 1340.3 |
| Mean completion tokens | 327.9 |
| Mean cost | $0.000000 |

In-process TestClient on CPU. No browser and no extra network hop. cost_usd is 0 for the deterministic provider because it does not call Groq. Token counts for that provider are size estimates (characters/4), not API usage.

Groq list prices used when a Groq run reports tokens (checked 2026-11-12 on the Groq models page): `openai/gpt-oss-20b` is $0.075 / 1M input tokens and $0.30 / 1M output tokens. The free tier bills $0. This run's cost column is the estimator in `app/agent/pricing.py`, which returns $0 for the deterministic provider.

## Pairs

| Id | Resume | Job | Label | Predicted | Score |
| --- | --- | --- | --- | --- | --- |
| p01 | avery_chen | backend_python | strong | strong | 93 |
| p02 | avery_chen | ai_engineer | maybe | maybe | 71 |
| p03 | avery_chen | platform | maybe | maybe | 64 |
| p04 | avery_chen | frontend | maybe | maybe | 57 |
| p05 | mina_okonkwo | backend_python | maybe | maybe | 68 |
| p06 | mina_okonkwo | ai_engineer | strong | strong | 93 |
| p07 | mina_okonkwo | platform | skip | skip | 21 |
| p08 | mina_okonkwo | frontend | skip | skip | 0 |
| p09 | luis_ortega | backend_python | skip | skip | 42 |
| p10 | luis_ortega | ai_engineer | skip | skip | 0 |
| p11 | luis_ortega | platform | strong | strong | 100 |
| p12 | luis_ortega | frontend | skip | skip | 0 |
| p13 | priya_shah | backend_python | skip | skip | 0 |
| p14 | priya_shah | ai_engineer | skip | skip | 0 |
| p15 | priya_shah | platform | skip | skip | 0 |
| p16 | priya_shah | frontend | strong | strong | 100 |
| p17 | samir_rahman | backend_python | skip | skip | 8 |
| p18 | samir_rahman | ai_engineer | skip | skip | 0 |
| p19 | samir_rahman | platform | skip | skip | 0 |
| p20 | samir_rahman | frontend | skip | skip | 0 |

## Paraphrase probes

- `probe_fastapi` k=5: FastAPI=yes. Query: Has this person built HTTP APIs in Python?
- `probe_docker` k=5: Docker=yes. Query: container images and local orchestration
- `probe_rag` k=5: retrieval-augmented=yes. Query: finding relevant passages before answering a question
- `probe_eval` k=5: evaluation=yes. Query: How did they measure whether model answers were any good?
- `probe_k8s` k=5: Kubernetes=yes. Query: orchestrating containers across a cluster
- `probe_go` k=5: Go=yes. Query: a statically typed language used for network services
- `probe_ui` k=5: React=yes, TypeScript=yes. Query: component-based interfaces in a typed frontend language
- `probe_css` k=5: CSS=yes. Query: layout and styling that survive a narrow phone screen
