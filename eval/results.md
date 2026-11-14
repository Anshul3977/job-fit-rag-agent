# Eval results

**Synthetic dataset.** 20 hand-labeled resume/job pairs (5 fictional resumes x 4 fictional job posts) in `eval/labels.json`, labeled before scoring with the rubric in that file. None of these people or openings are real. With N=20 and a lexical-friendly synthetic set, these numbers say the pipeline works end to end; they are not a claim about accuracy on real resumes.

- Date: 2026-11-15 (IST). Script: `python eval/run_eval.py` (in-process Starlette TestClient, CPU, no browser, no extra network hop).
- LLM run: `EVAL_USE_GROQ=1 LLM_PROVIDER=groq GROQ_MODEL=openai/gpt-oss-20b`, Groq free tier, temperature 0. Raw rows: [`results_groq.json`](results_groq.json).
- Baseline: deterministic scorer (`deterministic-v1`, no LLM), same script and data. Raw rows: [`results_deterministic.json`](results_deterministic.json) (also copied to `results.json`, the script's default output).
- Embeddings `fastembed` / `sentence-transformers/all-MiniLM-L6-v2`, vector store `chroma`. Machine: Linux-6.12.113+-x86_64-with-glibc2.41, 8 CPUs, Python 3.13.5.
- In both runs the final score and verdict are computed by the same code (`app/agent/scoring.py`) from per-requirement judgments; the LLM only proposes the requirement list, the judgments and quotes, and the summary.

## Metrics

| Metric | Groq `openai/gpt-oss-20b` | Deterministic baseline |
| --- | --- | --- |
| Pairs scored / failures | 20 / 0 | 20 / 0 |
| Verdict accuracy (exact) | 95.0% (19/20) | 100.0% (20/20) |
| Adjacent agreement (off-by-one = 0.5) | 97.5% | 100.0% |
| Majority-class baseline (always `skip`) | 60.0% | 60.0% |
| Required-skill hit rate | 100.0% | 100.0% |
| Unsupported-skill rejection rate | 98.0% (48/49) | 100.0% |
| Citation validity (quote is exact substring of cited chunk) | 100.0% | 100.0% |
| Lexical recall@5 (retrieval, same for both) | 100.0% | 100.0% |
| Paraphrase-probe recall@5 (retrieval, same for both) | 100.0% | 100.0% |
| Retrieve p50 | 25.0 ms | 18.0 ms |
| Fit latency p50 / p95 (full run, see caveat) | 31733.5 / 42245.1 ms | 23.5 / 29.1 ms |
| Mean prompt / completion tokens | 3028.1 / 1504.0 (Groq usage) | 1340.3 / 327.9 (chars/4 estimate) |
| Mean cost per fit (list-price estimate) | $0.000678 | $0 |
| Wall clock for the run | 628.17 s | 4.37 s |

### Latency caveat

The Groq free tier allows 8,000 tokens per minute for this model, and one fit uses about 4,500 tokens (3 calls). During the full 20-pair run the eval harness retried rate-limited (HTTP 429) Groq calls after Groq's `retry-after`, and those waits happened inside the request, so the full-run p50/p95 above (31.7 s / 42.2 s) mostly measure rate-limit waiting, not model speed. Only the first pair (p01, 3,257 ms) ran with a full token budget.

To measure latency without that wait, pairs p01-p05 were re-run on 2026-11-15 one at a time with a 62 s pause between fits, against a local `uvicorn` server (1 worker) with the same config, over HTTP on localhost. Raw rows: [`latency_sample_paced_groq.json`](latency_sample_paced_groq.json).

| Paced sample (N=5) | Value |
| --- | --- |
| Server fit latency min / median / max | 2645 / 3103 / 3629 ms |
| Verdicts | same as the full run for all 5 pairs (including the p02 miss) |

N=5 is too small for a p95. Without pacing, a second fit started right after the first can fail with HTTP 503 (`LLM_UNAVAILABLE`) because the app retries a 429 only once after 0.4 s.

## Errors

- LLM run: 1 of 20 verdicts wrong. **p02** (avery_chen x ai_engineer), labeled `maybe`, was scored `strong` 100 by Groq (the deterministic scorer gave `maybe` 71). It is also the one unsupported-skill rejection failure (0/1): the model marked a required skill the label says is missing as a match. This reproduced in the paced re-run.
- 18 of 20 pairs got the same score from both providers; p17 differs (Groq `skip` 0 vs deterministic `skip` 8) and p02 as above.
- Every citation the LLM returned was an exact substring of the cited chunk (100%). The pipeline's self-check step enforces this, so this rate reflects the guard as much as the model.

## Pairs

| Id | Resume | Job | Label | Groq verdict | Groq score | Groq citations valid | Deterministic verdict | Deterministic score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p01 | avery_chen | backend_python | strong | strong | 93 | 6/6 | strong | 93 |
| p02 | avery_chen | ai_engineer | maybe | strong | 100 | 6/6 | maybe | 71 |
| p03 | avery_chen | platform | maybe | maybe | 64 | 3/3 | maybe | 64 |
| p04 | avery_chen | frontend | maybe | maybe | 57 | 2/2 | maybe | 57 |
| p05 | mina_okonkwo | backend_python | maybe | maybe | 68 | 4/4 | maybe | 68 |
| p06 | mina_okonkwo | ai_engineer | strong | strong | 93 | 5/5 | strong | 93 |
| p07 | mina_okonkwo | platform | skip | skip | 21 | 1/1 | skip | 21 |
| p08 | mina_okonkwo | frontend | skip | skip | 0 | 0/0 | skip | 0 |
| p09 | luis_ortega | backend_python | skip | skip | 42 | 3/3 | skip | 42 |
| p10 | luis_ortega | ai_engineer | skip | skip | 0 | 0/0 | skip | 0 |
| p11 | luis_ortega | platform | strong | strong | 100 | 6/6 | strong | 100 |
| p12 | luis_ortega | frontend | skip | skip | 0 | 0/0 | skip | 0 |
| p13 | priya_shah | backend_python | skip | skip | 0 | 0/0 | skip | 0 |
| p14 | priya_shah | ai_engineer | skip | skip | 0 | 0/0 | skip | 0 |
| p15 | priya_shah | platform | skip | skip | 0 | 0/0 | skip | 0 |
| p16 | priya_shah | frontend | strong | strong | 100 | 5/5 | strong | 100 |
| p17 | samir_rahman | backend_python | skip | skip | 0 | 0/0 | skip | 8 |
| p18 | samir_rahman | ai_engineer | skip | skip | 0 | 0/0 | skip | 0 |
| p19 | samir_rahman | platform | skip | skip | 0 | 0/0 | skip | 0 |
| p20 | samir_rahman | frontend | skip | skip | 0 | 0/0 | skip | 0 |

## Paraphrase probes (retrieval only, identical in both runs)

- `probe_fastapi` k=5: FastAPI=yes. Query: Has this person built HTTP APIs in Python?
- `probe_docker` k=5: Docker=yes. Query: container images and local orchestration
- `probe_rag` k=5: retrieval-augmented=yes. Query: finding relevant passages before answering a question
- `probe_eval` k=5: evaluation=yes. Query: How did they measure whether model answers were any good?
- `probe_k8s` k=5: Kubernetes=yes. Query: orchestrating containers across a cluster
- `probe_go` k=5: Go=yes. Query: a statically typed language used for network services
- `probe_ui` k=5: React=yes, TypeScript=yes. Query: component-based interfaces in a typed frontend language
- `probe_css` k=5: CSS=yes. Query: layout and styling that survive a narrow phone screen

Cost note: Groq list price for `openai/gpt-oss-20b` (checked 2026-11-12) is $0.075 / 1M input and $0.30 / 1M output tokens; the free tier bills $0. The cost row is the estimator in `app/agent/pricing.py`.
