# Phase 4 A/B — compaction strategy recall

Model: `Qwen3-4B-Instruct-2507-Q6_K` · Embedder: `jina-embeddings-v5-small-retrieval` · ctx=8192 · seed=42
Session: 8 real DeepSeek-V4 paper chunks (~1500 tok), 4 facts planted at chunks 1/3/5/7
(F1 earliest → F4 latest), probed after all 8 chunks (so every fact is compacted first).
Recall = the fact's unique token appears in the model's answer (piped input is not echoed,
so a token in the output was generated, i.e. recalled). Reproduce: `tests/mem_bench/run.sh`.

## Run B — DISTINCT facts (fair)
Different topics, unique tokens (architect=ZORRILLA_7741, cluster=BLUEFIN_9931,
deadline=GLACIER_0414, budget=TUNGSTEN_4271) so embedding retrieval can disambiguate them.

| mode | F1 | F2 | F3 | F4 | recall | compactions | wall | crash |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| off      | ❌ | ❌ | ❌ | ❌ | 0/4 | 4 | 0 | — |
| summary  | ❌ | ❌ | ❌ | ❌ | 0/4 | 7 | 0 | — |
| retrieve | ❌ | ✅ | ❌ | ✅ | 2/4 | 7 | 0 | — |
| hybrid   | ✅ | ✅ | ❌ | ❌ | 2/4 | 9 | 0 | — |

**Retrieval (2/4) beats summary (0/4)** for fact recall — supports MemGPT/Mem0. Neither is
reliable: a unique token buried at the end of a 1500-token chunk, summarized by a 4B model or
matched by a small embedder, is a hard case. Summary's miss is compression (the needle is
lost); retrieval's misses are dilution (the fact shares its 900-char window with unrelated
paper text). 0 walls, 0 crashes across all modes.

## Run A — NEAR-IDENTICAL facts (biased; kept as a cautionary example)
v1 facts were "the access code for vault one/two/three/four is <token>" — near-duplicates.

| mode | recall | note |
|---|:--:|---|
| off      | 0/4 | baseline |
| summary  | 4/4 | the 4 near-identical lines form a list the LLM copies wholesale; the summary prompt is tuned for "codes to remember" |
| retrieve | 1/4 | the 4 windows embed almost identically → retrieval can't disambiguate "vault one" vs "two"; the model answered the same code (MARMOT/vault two) for everything |
| hybrid   | 4/4 | inherits summary's list |

## Takeaways
1. **Benchmark design decided the winner** — near-duplicate facts favour summary, distinct
   facts favour retrieval. Always measure with realistic, distinct facts.
2. Retrieval is the better *direction* for fact fidelity (model-agnostic, deterministic), but
   the v1 implementation is mediocre (2/4). Headroom: smaller/overlapping chunks, higher k,
   re-ranking, query rewriting.
3. Summary still preserves *gist* ("what was this about"); hybrid is the intended production
   combo but only pays off once retrieval is actually good.
4. Cost: retrieve/hybrid pay an embedding per dropped chunk + a query embedding per turn
   (cheap, deterministic) vs summary's one LLM generation per compaction.
