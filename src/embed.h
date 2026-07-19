#ifndef BASI_EMBED_H
#define BASI_EMBED_H

#include <stddef.h>
#include <stdbool.h>

/* Lazy-loaded embedding subsystem (step 10 of the knowledge-base milestone).
 *
 * Loads a separate llama_model + llama_context dedicated to embedding
 * inference. The model is resolved via:
 *   1. $BASI_EMBED_MODEL (absolute path to a GGUF), then
 *   2. ~/.cache/llama.cpp/<glob: ...jina...v5...retrieval...gguf> (default
 *      download location for `llama-... -hf jinaai/jina-embeddings-v5-text-
 *      small-retrieval-GGUF`), then
 *   3. ~/.cache/llama.cpp/<glob: ...embed...gguf> (any embedding model
 *      present).
 *
 * Pooling is LAST-token (matches Jina v5 / Qwen3-Embedding family). If you
 * use BGE-M3 or another mean-pooled model, set BASI_EMBED_POOLING=mean. */

/* One-shot lazy init. 0 on success, -1 on failure (see embed_last_error()).
 * Safe to call repeatedly; subsequent calls are no-ops once loaded. */
int  embed_init(void);

/* True if an embedding model can be located (BASI_EMBED_MODEL, ~/.cache/llama.cpp,
 * or the HF hub cache) WITHOUT loading it — a cheap pre-flight so a caller can warn
 * and fall back before relying on embeddings. Sets embed_last_error() on false. */
bool embed_available(void);

/* Embedding dimension after init (1024 for Jina v5, 768 for EmbeddingGemma).
 * Returns -1 before init. */
int  embed_dim(void);

/* Embed `n` texts in ONE request. Writes row i at out[i * embed_dim()], so `out`
 * must hold n * embed_dim() floats. Returns 0 on success, -1 on failure (all or
 * nothing — a partial batch is reported as failure rather than leaving stale
 * rows, which would corrupt retrieval silently). Rows are L2-normalized exactly
 * as embed_text does, so batch and single results are interchangeable in a store
 * (verified: worst cosine(single,batch) = 0.999994, i.e. float rounding only).
 * ~2.2x faster than looping embed_text, plateauing by N~64 at ~19 ms/chunk. */
int  embed_texts(const char **texts, int n, float *out);

/* Self-test (BASI_EMBED_BATCH_SELFTEST=1): batch-vs-single equivalence + speedup. */
void embed_batch_selftest(void);

/* Embed `text` into `out` (caller-allocated, must be at least
 * embed_dim() floats). Output is L2-normalized so dot product == cosine
 * similarity. Returns 0 on success, -1 on failure. */
int  embed_text(const char *text, float *out);

/* Free the embed model + context. Idempotent. */
void embed_shutdown(void);

/* Most recent error from embed_init / embed_text. Never NULL. */
const char *embed_last_error(void);

/* Tool entry point: `args` is the user query.
 * Lazy-inits the model, syncs the on-disk vector store with the current
 * KB state (mtime-based invalidation), embeds the query, returns the top
 * matches via dot product on L2-normalized vectors. */
char *execute_docs_vector_search(const char *args);

#endif /* BASI_EMBED_H */
