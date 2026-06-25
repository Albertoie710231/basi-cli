/* Session-scoped retrieval memory (Phase 4). See memory.h. */
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>

#include "memory.h"
#include "embed.h"

#define MEM_DBG(...) do { if (getenv("BASI_DEBUG_RECLAIM")) fprintf(stderr, __VA_ARGS__); } while (0)

/* Store retrieval units at SENTENCE granularity, not fixed windows, so a single
 * fact ("...the lead architect is ZORRILLA_7741.") becomes its own clean memory
 * instead of being diluted inside a wide window of unrelated text. This is the
 * deterministic stand-in for Mem0-style atomic-fact extraction: each stored unit
 * is 1+ sentences up to MEM_CHUNK_MAX chars; sentences shorter than MEM_CHUNK_MIN
 * are merged forward so trivial fragments ("OK.") don't become their own vector. */
#define MEM_CHUNK_MAX 320
#define MEM_CHUNK_MIN 40

typedef struct { float *vec; char *text; } MemChunk;

static MemChunk *g_chunks = NULL;
static size_t    g_n      = 0;
static size_t    g_cap    = 0;
static int       g_dim    = 0;

static bool blank(const char *s) {
    if (!s) return true;
    for (; *s; s++) if (!isspace((unsigned char)*s)) return false;
    return true;
}

/* Embed + store one already-sized window. */
static void mem_add_one(const char *text) {
    if (blank(text)) return;
    if (embed_init() != 0) { MEM_DBG("[mem] embed_init failed: %s\n", embed_last_error()); return; }
    int dim = embed_dim();
    if (dim <= 0) { MEM_DBG("[mem] embed_dim=%d\n", dim); return; }
    g_dim = dim;

    float *v = malloc((size_t)dim * sizeof(float));
    if (!v) return;
    if (embed_text(text, v) != 0) { MEM_DBG("[mem] embed_text failed: %s\n", embed_last_error()); free(v); return; }

    if (g_n == g_cap) {
        size_t nc = g_cap ? g_cap * 2 : 16;
        MemChunk *nn = realloc(g_chunks, nc * sizeof(MemChunk));
        if (!nn) { free(v); return; }
        g_chunks = nn; g_cap = nc;
    }
    char *copy = strdup(text);
    if (!copy) { free(v); return; }
    g_chunks[g_n].vec  = v;
    g_chunks[g_n].text = copy;
    g_n++;
}

void mem_add(const char *text) {
    if (blank(text)) return;
    size_t len = strlen(text);

    size_t i = 0;
    while (i < len) {
        while (i < len && isspace((unsigned char)text[i])) i++;   /* skip leading ws */
        if (i >= len) break;
        size_t start = i, end = start;
        while (end < len) {
            char c = text[end];
            end++;
            /* a real sentence end is . ! ? followed by space/EOF (so "1.6T" and
               "e.g." don't split); newline also ends a unit. */
            bool sentence_end = (c == '.' || c == '!' || c == '?') &&
                                (end >= len || isspace((unsigned char)text[end]));
            if ((sentence_end || c == '\n') && (end - start) >= MEM_CHUNK_MIN) break;
            if (end - start >= MEM_CHUNK_MAX) {           /* hard cap: back off to a space */
                size_t e = end;
                while (e > start && !isspace((unsigned char)text[e])) e--;
                if (e > start + MEM_CHUNK_MAX / 2) end = e;
                break;
            }
        }
        size_t sl = end - start;
        while (sl > 0 && isspace((unsigned char)text[start + sl - 1])) sl--;   /* trim */
        char *piece = malloc(sl + 1);
        if (piece) {
            memcpy(piece, text + start, sl);
            piece[sl] = '\0';
            if (!blank(piece)) mem_add_one(piece);
            free(piece);
        }
        i = end;
    }
}

int mem_retrieve(const char *query, int k, float threshold,
                 char **out_texts, float *out_scores) {
    if (g_n == 0 || k <= 0 || !out_texts || blank(query)) return 0;
    if (embed_init() != 0) return 0;
    int dim = embed_dim();
    if (dim <= 0 || (g_dim && dim != g_dim)) return 0;

    float *q = malloc((size_t)dim * sizeof(float));
    if (!q) return 0;
    if (embed_text(query, q) != 0) { free(q); return 0; }

    float *scores = malloc(g_n * sizeof(float));
    if (!scores) { free(q); return 0; }
    for (size_t i = 0; i < g_n; i++) {
        float s = 0.0f;
        const float *v = g_chunks[i].vec;
        for (int d = 0; d < dim; d++) s += q[d] * v[d];   /* cosine (L2-normalized) */
        scores[i] = s;
    }
    free(q);

    /* greedy top-k above threshold (g_n is small — a session's dropped turns) */
    char *used = calloc(g_n, 1);
    if (!used) { free(scores); return 0; }
    int out = 0;
    for (int slot = 0; slot < k; slot++) {
        int best = -1;
        float best_s = threshold;
        for (size_t i = 0; i < g_n; i++) {
            if (used[i]) continue;
            if (scores[i] >= best_s) { best_s = scores[i]; best = (int)i; }
        }
        if (best < 0) break;            /* nothing left above threshold */
        used[best] = 1;
        char *copy = strdup(g_chunks[best].text);
        if (!copy) break;
        out_texts[out] = copy;
        if (out_scores) out_scores[out] = scores[best];
        out++;
    }
    free(used);
    free(scores);
    return out;
}

size_t mem_count(void) { return g_n; }

void mem_clear(void) {
    for (size_t i = 0; i < g_n; i++) { free(g_chunks[i].vec); free(g_chunks[i].text); }
    free(g_chunks);
    g_chunks = NULL; g_n = 0; g_cap = 0;
}
