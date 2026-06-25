#ifndef BASI_MEMORY_H
#define BASI_MEMORY_H

#include <stddef.h>

/* Session-scoped retrieval memory (Phase 4). Stores embedded chunks of dropped
 * conversation turns and retrieves the top-k most similar to a query. Uses the
 * existing SEPARATE embedder (embed.h), so recall quality is independent of the
 * chat model. Deterministic given the embeddings (cosine via dot product on
 * L2-normalized vectors). */

/* Embed `text` and append it to the index (stores its own copy). No-op on empty/
 * whitespace text or embedding failure. */
void mem_add(const char *text);

/* Retrieve up to `k` stored chunks with cosine score >= `threshold`, best first.
 * out_texts[i] receives a malloc'd copy (caller frees each); out_scores may be
 * NULL. Returns the count written (0..k). */
int mem_retrieve(const char *query, int k, float threshold,
                 char **out_texts, float *out_scores);

size_t mem_count(void);
void   mem_clear(void);   /* free the whole index */

#endif /* BASI_MEMORY_H */
