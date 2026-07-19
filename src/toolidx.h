#ifndef BASI_TOOLIDX_H
#define BASI_TOOLIDX_H
/* Tool-result index — the "retrieve, don't stuff" store.
 *
 * A research turn pastes whole tool results into context: measured on a
 * 21-tool-call run, ~70% of wall-clock was PREFILLING tool output (35k prefill
 * tokens at ~40 t/s), and generation decayed 27.0 -> 18.8 t/s as context grew.
 * Worse, truncate_tool_result() caps results at 8000 bytes and DESTROYS the
 * middle — logs show `2 lines · trimmed 237k`. So today BASI both pays for the
 * tokens it keeps and throws away the ones it doesn't.
 *
 * This store keeps the whole result addressable: index it once, inject only the
 * chunks relevant to the task, and let the model pull more by id.
 *
 * WHY NOT memory.c: (1) /clear calls mem_clear() (main.c) and would nuke tool
 * results along with conversation memory; (2) mixing tool output into the same
 * corpus that feeds per-turn retrieval injection changes that feature's quality;
 * (3) memory.c is all-globals with no room for provenance fields. It is also the
 * wrong CHUNKER — mem_add splits sentence-atomically at 40..320 chars, which is
 * meaningless for HTML, grep dumps and code, and far too fine: at the measured
 * ~19 ms/chunk floor, a 237 KB page at 200-char chunks costs ~23 s to index
 * against a ~37 s/turn prefill saving. That barely breaks even, so this chunker
 * is COARSE and block-oriented (see TOOLIDX_CHUNK_TARGET). */
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Target chunk size in bytes. Coarse on purpose — see the cost note above. */
#define TOOLIDX_CHUNK_TARGET 2048
/* Results smaller than this are cheap to paste verbatim; indexing them would pay
 * an embed round trip to save nothing. */
#define TOOLIDX_MIN_INDEX    4096

/* Index one tool result. `tool` and `args` are provenance shown to the model when
 * a chunk is retrieved, so it knows WHERE a fragment came from. Returns the number
 * of chunks stored (0 if skipped or on failure — always non-fatal: a failure here
 * must degrade to the existing paste-and-truncate path, never break the turn).
 * The result id for later addressing is written to *out_result_id. */
int toolidx_add(const char *tool, const char *args, const char *text, int *out_result_id);

/* Retrieve up to k chunks matching `query`, best first. Returns the count written.
 * out_texts[i] are borrowed pointers valid until toolidx_clear(); out_ids[i] is the
 * global chunk id for addressing. Either out array may be NULL. */
int toolidx_retrieve(const char *query, int k, float threshold,
                     const char **out_texts, int *out_ids, float *out_scores);

/* Fetch one chunk verbatim by id (borrowed, NULL if unknown) — backs the tool that
 * lets the model say "give me chunk 37" after seeing a neighbouring fragment. */
const char *toolidx_get(int chunk_id);

/* Chunks stored for one result id, and the id range, so a caller can describe what
 * is addressable ("chunks 12-38 of result 3"). Returns 0 if the id is unknown. */
int toolidx_result_span(int result_id, int *out_first_id, int *out_last_id);

/* Self-test (BASI_TOOLIDX_SELFTEST=1): index a large synthetic result with a
 * buried needle and check retrieval surfaces it. Needs a live embedder. */
void   toolidx_selftest(void);

size_t toolidx_count(void);
void   toolidx_clear(void);

#ifdef __cplusplus
}
#endif
#endif /* BASI_TOOLIDX_H */
