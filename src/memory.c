/* Session-scoped retrieval memory (Phase 4). See memory.h.
 *
 * Scoring is selectable via BASI_RETRIEVE_SCORE:
 *   dense  (default) — cosine over embeddings (semantic; fuzzy on exact tokens)
 *   bm25            — sparse keyword scoring (pure code, no model; exact-token strong)
 *   hybrid          — reciprocal-rank fusion of dense + bm25 (complementary failure modes)
 * Whether hybrid actually beats dense in this local/small-model regime is an empirical
 * question — that's what tests/mem_bench measures. */
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "memory.h"
#include "embed.h"

#define MEM_DBG(...) do { if (getenv("BASI_DEBUG_RECLAIM")) fprintf(stderr, __VA_ARGS__); } while (0)

/* Sentence-granularity chunks (deterministic stand-in for atomic-fact extraction):
   each stored unit is 1+ sentences up to MEM_CHUNK_MAX chars; fragments shorter than
   MEM_CHUNK_MIN merge forward so "OK." doesn't become its own memory. */
#define MEM_CHUNK_MAX 320
#define MEM_CHUNK_MIN 40

typedef struct {
    float    *vec;      /* dense embedding (NULL if no embedder) */
    char     *text;     /* verbatim chunk */
    uint32_t *terms;    /* hashed lowercased alnum tokens (for BM25) */
    int       n_terms;
} MemChunk;

static MemChunk *g_chunks = NULL;
static size_t    g_n   = 0;
static size_t    g_cap = 0;
static int       g_dim = 0;

static bool blank(const char *s) {
    if (!s) return true;
    for (; *s; s++) if (!isspace((unsigned char)*s)) return false;
    return true;
}

/* Tokenize into FNV-1a hashes of lowercased alphanumeric runs (BM25 terms). */
static uint32_t *tokenize_terms(const char *text, int *n_out) {
    int cap = 16, n = 0;
    uint32_t *t = malloc((size_t)cap * sizeof(uint32_t));
    if (!t) { *n_out = 0; return NULL; }
    const unsigned char *p = (const unsigned char *)text;
    while (*p) {
        while (*p && !isalnum(*p)) p++;
        if (!*p) break;
        uint32_t h = 2166136261u;
        while (*p && isalnum(*p)) { h ^= (unsigned char)tolower(*p); h *= 16777619u; p++; }
        if (n == cap) { cap *= 2; uint32_t *nn = realloc(t, (size_t)cap * sizeof(uint32_t)); if (!nn) break; t = nn; }
        t[n++] = h;
    }
    *n_out = n;
    return t;
}

static int term_tf(const MemChunk *c, uint32_t h) {
    int tf = 0;
    for (int j = 0; j < c->n_terms; j++) if (c->terms[j] == h) tf++;
    return tf;
}

/* Embed (best-effort) + tokenize + store one window. Tolerates a missing embedder
   (vec=NULL) so BM25-only scoring still works. */
static void mem_add_one(const char *text) {
    if (blank(text)) return;
    float *v = NULL;
    if (embed_init() == 0) {
        int dim = embed_dim();
        if (dim > 0) {
            g_dim = dim;
            v = malloc((size_t)dim * sizeof(float));
            if (v && embed_text(text, v) != 0) {
                MEM_DBG("[mem] embed_text failed: %s\n", embed_last_error());
                free(v); v = NULL;
            }
        }
    } else {
        MEM_DBG("[mem] embed_init failed: %s\n", embed_last_error());
    }

    int nt = 0;
    uint32_t *terms = tokenize_terms(text, &nt);
    char *copy = strdup(text);
    if (!copy) { free(v); free(terms); return; }

    if (g_n == g_cap) {
        size_t nc = g_cap ? g_cap * 2 : 16;
        MemChunk *nn = realloc(g_chunks, nc * sizeof(MemChunk));
        if (!nn) { free(v); free(terms); free(copy); return; }
        g_chunks = nn; g_cap = nc;
    }
    g_chunks[g_n].vec     = v;
    g_chunks[g_n].text    = copy;
    g_chunks[g_n].terms   = terms;
    g_chunks[g_n].n_terms = nt;
    g_n++;
}

void mem_add(const char *text) {
    if (blank(text)) return;
    size_t len = strlen(text);
    size_t i = 0;
    while (i < len) {
        while (i < len && isspace((unsigned char)text[i])) i++;
        if (i >= len) break;
        size_t start = i, end = start;
        while (end < len) {
            char c = text[end];
            end++;
            bool sentence_end = (c == '.' || c == '!' || c == '?') &&
                                (end >= len || isspace((unsigned char)text[end]));
            if ((sentence_end || c == '\n') && (end - start) >= MEM_CHUNK_MIN) break;
            if (end - start >= MEM_CHUNK_MAX) {
                size_t e = end;
                while (e > start && !isspace((unsigned char)text[e])) e--;
                if (e > start + MEM_CHUNK_MAX / 2) end = e;
                break;
            }
        }
        size_t sl = end - start;
        while (sl > 0 && isspace((unsigned char)text[start + sl - 1])) sl--;
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

typedef enum { SC_DENSE, SC_BM25, SC_HYBRID } ScoreMode;
static ScoreMode score_mode(void) {
    const char *s = getenv("BASI_RETRIEVE_SCORE");
    if (s) { if (!strcmp(s, "bm25")) return SC_BM25; if (!strcmp(s, "hybrid")) return SC_HYBRID; }
    return SC_DENSE;
}

/* dense cosine for every chunk (vec-less chunks get a sentinel low score). */
static double *dense_scores(const char *query) {
    if (embed_init() != 0) return NULL;
    int dim = embed_dim();
    if (dim <= 0 || (g_dim && dim != g_dim)) return NULL;
    float *q = malloc((size_t)dim * sizeof(float));
    if (!q) return NULL;
    if (embed_text(query, q) != 0) { free(q); return NULL; }
    double *s = malloc(g_n * sizeof(double));
    if (s) for (size_t i = 0; i < g_n; i++) {
        const float *v = g_chunks[i].vec;
        if (!v) { s[i] = -1e9; continue; }
        double d = 0; for (int j = 0; j < dim; j++) d += (double)q[j] * v[j];
        s[i] = d;
    }
    free(q);
    return s;
}

/* BM25 over the session corpus (k1=1.2, b=0.75). */
static double *bm25_scores(const char *query) {
    int nq = 0;
    uint32_t *qt = tokenize_terms(query, &nq);
    if (!qt || nq == 0) { free(qt); return NULL; }
    double *s = calloc(g_n, sizeof(double));
    if (!s) { free(qt); return NULL; }
    double avgdl = 0;
    for (size_t i = 0; i < g_n; i++) avgdl += g_chunks[i].n_terms;
    avgdl = avgdl > 0 ? avgdl / (double)g_n : 1.0;
    const double k1 = 1.2, b = 0.75;
    for (int a = 0; a < nq; a++) {
        bool dup = false;
        for (int z = 0; z < a; z++) if (qt[z] == qt[a]) { dup = true; break; }
        if (dup) continue;
        int df = 0;
        for (size_t i = 0; i < g_n; i++) if (term_tf(&g_chunks[i], qt[a]) > 0) df++;
        if (df == 0) continue;
        double idf = log(1.0 + ((double)g_n - df + 0.5) / ((double)df + 0.5));
        for (size_t i = 0; i < g_n; i++) {
            int tf = term_tf(&g_chunks[i], qt[a]);
            if (!tf) continue;
            double denom = tf + k1 * (1.0 - b + b * (double)g_chunks[i].n_terms / avgdl);
            s[i] += idf * (tf * (k1 + 1.0)) / denom;
        }
    }
    free(qt);
    return s;
}

int mem_retrieve(const char *query, int k, float threshold,
                 char **out_texts, float *out_scores) {
    if (g_n == 0 || k <= 0 || !out_texts || blank(query)) return 0;
    ScoreMode mode = score_mode();
    size_t N = g_n;

    double *dense = (mode == SC_DENSE || mode == SC_HYBRID) ? dense_scores(query) : NULL;
    double *bm25  = (mode == SC_BM25  || mode == SC_HYBRID) ? bm25_scores(query)  : NULL;
    if (mode == SC_DENSE && !dense) { return 0; }
    if (mode == SC_BM25  && !bm25)  { free(dense); return 0; }

    double *final = malloc(N * sizeof(double));
    char   *elig  = calloc(N, 1);
    if (!final || !elig) { free(final); free(elig); free(dense); free(bm25); return 0; }

    if (mode == SC_DENSE) {
        for (size_t i = 0; i < N; i++) { final[i] = dense[i]; elig[i] = dense[i] >= threshold; }
    } else if (mode == SC_BM25) {
        for (size_t i = 0; i < N; i++) { final[i] = bm25[i]; elig[i] = bm25[i] > 0.0; }
    } else { /* hybrid: RRF over chunks eligible in either signal (ranks via O(N^2), N small) */
        for (size_t i = 0; i < N; i++)
            elig[i] = (dense && dense[i] >= threshold) || (bm25 && bm25[i] > 0.0);
        for (size_t i = 0; i < N; i++) {
            int dr = 0, br = 0;
            for (size_t j = 0; j < N; j++) {
                if (dense && (dense[j] > dense[i] || (dense[j] == dense[i] && j < i))) dr++;
                if (bm25  && (bm25[j]  > bm25[i]  || (bm25[j]  == bm25[i]  && j < i))) br++;
            }
            double rrf = 0.0;
            if (dense) rrf += 1.0 / (60.0 + dr);
            if (bm25)  rrf += 1.0 / (60.0 + br);
            final[i] = rrf;
        }
    }

    int out = 0;
    char *used = calloc(N, 1);
    if (!used) { free(final); free(elig); free(dense); free(bm25); return 0; }
    for (int slot = 0; slot < k; slot++) {
        int best = -1; double best_s = 0;
        for (size_t i = 0; i < N; i++) {
            if (used[i] || !elig[i]) continue;
            if (best < 0 || final[i] > best_s) { best_s = final[i]; best = (int)i; }
        }
        if (best < 0) break;
        used[best] = 1;
        char *cpy = strdup(g_chunks[best].text);
        if (!cpy) break;
        out_texts[out] = cpy;
        if (out_scores) out_scores[out] = (float)final[best];
        out++;
    }
    if (getenv("BASI_DEBUG_RECLAIM"))
        fprintf(stderr, "[retrieve-score] mode=%s n=%zu returned=%d\n",
                mode == SC_BM25 ? "bm25" : mode == SC_HYBRID ? "hybrid" : "dense", N, out);

    free(used); free(final); free(elig); free(dense); free(bm25);
    return out;
}

size_t mem_count(void) { return g_n; }

void mem_clear(void) {
    for (size_t i = 0; i < g_n; i++) {
        free(g_chunks[i].vec);
        free(g_chunks[i].text);
        free(g_chunks[i].terms);
    }
    free(g_chunks);
    g_chunks = NULL; g_n = 0; g_cap = 0;
}
