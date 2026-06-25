# Retrieval scoring comparison (dense vs bm25 vs hybrid)

Model: `Qwen3-4B-Instruct-2507-Q6_K` · Embedder: `jina-embeddings-v5-small-retrieval` · ctx=8192 · seed=42 · mode=retrieve
Recall over 4 facts planted at chunks 1/3/5/7, probed after all are compacted.
Reproduce: `tests/mem_bench/score_bench.sh`. Select scoring with `BASI_RETRIEVE_SCORE=dense|bm25|hybrid`.

| fact regime | score | F1 | F2 | F3 | F4 | recall |
|---|---|:--:|:--:|:--:|:--:|:--:|
| distinct | dense  | ✅ | ✅ | ✅ | ✅ | 4/4 |
| distinct | bm25   | ✅ | ✅ | ✅ | ✅ | 4/4 |
| distinct | hybrid | ✅ | ✅ | ✅ | ✅ | 4/4 |
| neardup  | dense  | ✅ | ✅ | ✅ | ✅ | 4/4 |
| neardup  | bm25   | ✅ | ✅ | ✅ | ❌ | 3/4 |
| neardup  | hybrid | ✅ | ✅ | ✅ | ✅ | 4/4 |

## Finding: in this regime, scoring method doesn't matter — chunking does
- **Distinct facts:** dense = bm25 = hybrid = 4/4. The hybrid/BM25 gains reported in the
  RAG literature (strong on big, messy corpora) **do not transfer** to a small local model
  + small session corpus. (Confirms the skeptic's prior.)
- **Near-duplicates:** dense already gets **4/4**, and BM25 is **worse (3/4)** — the exact
  case where BM25 was "supposed" to win. The earlier "dense 1/4 on near-dup" failure was a
  **chunking artifact** (900-char windows blended "vault one"/"vault two"); sentence-level
  atomic chunks fixed it, and dense disambiguates them fine.
- **Hybrid never beats dense** — only matches it — while adding a whole sparse-index path.

**The lever was chunking granularity, not the scoring algorithm.** Once memories are atomic,
plain dense cosine suffices on a small corpus; BM25's exact-match edge addresses a dilution
problem that good chunking already removed.

## Decision
- Keep **`dense`** the default (measured equal-or-better, simplest).
- Keep **`bm25`/`hybrid`** available via `BASI_RETRIEVE_SCORE` for a future large-corpus
  regime where dense may degrade — but do NOT pay their complexity by default.
- Caveat: the 4-fact benchmark hits a 4/4 ceiling for dense, so it can't expose a scoring
  difference even if one existed at scale. A definitive hybrid test would need a much larger,
  noisier corpus — which is a different question than BASI's session-scale memory.
