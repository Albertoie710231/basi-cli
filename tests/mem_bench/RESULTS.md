# Phase 4 A/B — compaction strategy recall

Model: `Qwen3-4B-Instruct-2507-Q6_K` · Embedder: `jina-embeddings-v5-small-retrieval` · ctx=8192 · seed=42
Session: 8 real DeepSeek-V4 paper chunks (~1500 tok), 4 facts planted at chunks 1/3/5/7
(F1 earliest → F4 latest), probed after all 8 chunks (every fact is compacted first).
Recall = the fact's unique token appears in the model's answer (piped input isn't echoed,
so a token in output = recalled). Reproduce: `tests/mem_bench/run.sh` (writes `last_run.md`).

## Headline (distinct facts)

| mode | retrieval v1 (raw 900-char windows) | retrieval v2 (sentence-level chunks) |
|---|:--:|:--:|
| off      | 0/4 | 0/4 |
| summary  | 0/4 | 0/4 |
| retrieve | 2/4 | **4/4** |
| hybrid   | 2/4 | **4/4** |

**Tuned deterministic retrieval beats LLM summary 4/4 vs 0/4 on fact recall**, both keep gist,
0 walls / 0 crashes. The jump from 2/4 → 4/4 came from one change: store memories at
**sentence (atomic) granularity** instead of fixed 900-char windows.

### Speed (wall-clock for the full 13-turn session, incl. model load)

| mode | recall | time | compactions | note |
|---|:--:|:--:|:--:|---|
| off      | 0/4 | 36s  | 4 | floor: load + turns, no memory work |
| summary  | 0/4 | 343s | 7 | an LLM generation per compaction (~44s each) + compacts more often |
| retrieve | 4/4 | 89s  | 4 | cheap embeddings + 2nd-model load; fewer compactions |
| hybrid   | 4/4 | 462s | 8 | pays for both |

**retrieve is ~3.9× faster than summary AND more accurate** — summary's LLM-generation-per-
compaction is the dominant cost; retrieve only embeds (and compacts less since no summary
checkpoint inflates the context). hybrid is slowest. (Arc Vulkan, single GPU; absolute times
are backend-bound, but the *ratio* is the structural result.)

## Why — from the Mem0 paper (arXiv 2504.19413)
- Fixed-length raw-chunk RAG is the *baseline* Mem0 beats (~26% LLM-judge); the win comes from
  **LLM-extracted atomic facts**, embedded one per memory.
- A fact embedded inside a wide window of unrelated text is a *blended* vector → weak match
  (dilution). Our v1 (900-char windows) hit exactly this: 2/4, misses by dilution.
- **Sentence-level chunking** is the deterministic stand-in for atomic-fact extraction: each
  fact-sentence ("…the lead architect is ZORRILLA_7741.") becomes its own clean memory →
  high-similarity match → 4/4. No LLM, model-agnostic, fits the deterministic-first motto.

## Cautionary run — NEAR-IDENTICAL facts (why benchmark design matters)
v1 facts were "the access code for vault one/two/three/four is <token>" (near-duplicates):
summary 4/4, retrieve 1/4. Near-duplicate windows embed alike → retrieval can't disambiguate
(it answered the same code for every vault), while the LLM copies the obvious list. **Change
the facts and the winner flips** — always measure with realistic, distinct facts.

## Takeaways
1. Retrieval is the right direction for fact fidelity, and once memories are atomic it is
   **decisive** (4/4 vs 0/4) — model-agnostic and deterministic.
2. Granularity is everything: atomic/sentence units, not wide windows.
3. Summary still preserves gist; **hybrid** = retrieval (facts) + summary (gist) and matched
   retrieve here (4/4) — the safe production default once retrieval is good.
4. Cost: retrieve/hybrid pay an embedding per dropped sentence + a query embedding per turn
   (cheap, deterministic) vs summary's one LLM generation per compaction. retrieve also did
   FEWER compactions (4 vs 7) — no summary checkpoint growing the context.
5. Next headroom (not yet done): top-k/threshold tuning, overlap, a dedup/update step (Mem0's),
   and an optional LLM atomic-extraction tier for messy multi-fact sentences.
