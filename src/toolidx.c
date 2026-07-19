/* Tool-result index. See toolidx.h for why this is separate from memory.c and
 * why the chunker is coarse. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

#include "toolidx.h"
#include "embed.h"

/* Embedding batch size. Measured speedup plateaus by ~64 (2.1x) and gains almost
 * nothing at 256, so batch in 64s rather than building one enormous request. */
#define TI_BATCH 64

typedef struct {
    char  *text;       /* verbatim chunk (owned) */
    float *vec;        /* embedding, or NULL if the embedder was unavailable */
    int    result_id;
} TiChunk;

typedef struct {
    char *tool;        /* provenance: which tool produced this (owned) */
    char *args;        /* ...with which arguments (owned, may be NULL) */
    int   first_chunk;
    int   n_chunks;
} TiResult;

static TiChunk  *g_chunks = NULL;
static size_t    g_n = 0, g_cap = 0;
static TiResult *g_results = NULL;
static size_t    g_rn = 0, g_rcap = 0;
static int       g_dim = 0;

static bool ti_blank(const char *s, size_t len) {
    for (size_t i = 0; i < len; i++) if (!isspace((unsigned char) s[i])) return false;
    return true;
}

/* Coarse, block-oriented split. Tool output is lines — HTML, grep hits, code,
 * log records — not prose, so break on LINE boundaries and prefer a blank line
 * (a real block boundary) when one is near the target. Never split mid-line
 * unless a single line exceeds the target, and even then snap off UTF-8
 * continuation bytes so a chunk is always valid UTF-8 (the same mistake that
 * silently killed turns via truncate_tool_result). Returns malloc'd array of
 * malloc'd strings; *out_n gets the count. */
static char **ti_chunk(const char *text, size_t len, int *out_n) {
    size_t cap = len / TOOLIDX_CHUNK_TARGET + 4;
    char **out = calloc(cap, sizeof *out);
    if (!out) { *out_n = 0; return NULL; }

    int n = 0;
    size_t i = 0;
    while (i < len) {
        size_t start = i;
        size_t end   = start;
        size_t last_nl = 0, last_blank = 0;

        while (end < len) {
            const char *nl = memchr(text + end, '\n', len - end);
            size_t line_end = nl ? (size_t) (nl - text) + 1 : len;
            /* A blank line means the NEXT chunk can start at a real boundary. */
            if (nl && line_end - end <= 2 && ti_blank(text + end, line_end - end))
                last_blank = line_end;
            last_nl = line_end;
            if (line_end - start >= TOOLIDX_CHUNK_TARGET) { end = line_end; break; }
            end = line_end;
            if (!nl) break;
        }

        /* Prefer a blank-line break if it kept at least half the target. */
        if (last_blank > start && last_blank - start >= TOOLIDX_CHUNK_TARGET / 2)
            end = last_blank;
        else if (last_nl > start && end - start > TOOLIDX_CHUNK_TARGET)
            end = last_nl;

        if (end <= start) end = len;                     /* no progress guard */
        if (end - start > TOOLIDX_CHUNK_TARGET * 2) {    /* one gigantic line */
            end = start + TOOLIDX_CHUNK_TARGET;
            while (end > start && ((unsigned char) text[end] & 0xC0) == 0x80) end--;
        }

        size_t sl = end - start;
        while (sl > 0 && isspace((unsigned char) text[start + sl - 1])) sl--;
        if (sl > 0 && !ti_blank(text + start, sl)) {
            if ((size_t) n >= cap) {
                size_t nc = cap * 2;
                char **t = realloc(out, nc * sizeof *t);
                if (!t) break;
                memset(t + cap, 0, (nc - cap) * sizeof *t);
                out = t; cap = nc;
            }
            char *piece = malloc(sl + 1);
            if (piece) { memcpy(piece, text + start, sl); piece[sl] = 0; out[n++] = piece; }
        }
        i = end;
    }
    *out_n = n;
    return out;
}

static bool ti_grow_chunks(size_t need) {
    if (g_n + need <= g_cap) return true;
    size_t nc = g_cap ? g_cap : 32;
    while (nc < g_n + need) nc *= 2;
    TiChunk *t = realloc(g_chunks, nc * sizeof *t);
    if (!t) return false;
    memset(t + g_cap, 0, (nc - g_cap) * sizeof *t);
    g_chunks = t; g_cap = nc;
    return true;
}

int toolidx_add(const char *tool, const char *args, const char *text, int *out_result_id) {
    if (out_result_id) *out_result_id = -1;
    if (!text) return 0;
    size_t len = strlen(text);
    if (len < TOOLIDX_MIN_INDEX) return 0;      /* too small to be worth a round trip */

    int nch = 0;
    char **pieces = ti_chunk(text, len, &nch);
    if (!pieces || nch <= 0) { free(pieces); return 0; }

    /* Embedding is best-effort: without it the chunks are still ADDRESSABLE by id,
       which keeps the "fetch more" path working even when the embedder is down. */
    float *vecs = NULL;
    if (embed_init() == 0) {
        g_dim = embed_dim();
        if (g_dim > 0) {
            vecs = calloc((size_t) nch * g_dim, sizeof(float));
            if (vecs) {
                for (int off = 0; off < nch; off += TI_BATCH) {
                    int m = nch - off; if (m > TI_BATCH) m = TI_BATCH;
                    if (embed_texts((const char **) (pieces + off), m,
                                    vecs + (size_t) off * g_dim) != 0) {
                        free(vecs); vecs = NULL; break;   /* all-or-nothing per header */
                    }
                }
            }
        }
    }

    if (!ti_grow_chunks((size_t) nch)) {
        for (int i = 0; i < nch; i++) free(pieces[i]);
        free(pieces); free(vecs);
        return 0;
    }
    if (g_rn + 1 > g_rcap) {
        size_t nc = g_rcap ? g_rcap * 2 : 8;
        TiResult *t = realloc(g_results, nc * sizeof *t);
        if (!t) { for (int i = 0; i < nch; i++) free(pieces[i]); free(pieces); free(vecs); return 0; }
        memset(t + g_rcap, 0, (nc - g_rcap) * sizeof *t);
        g_results = t; g_rcap = nc;
    }

    int rid = (int) g_rn;
    g_results[g_rn].tool        = tool ? strdup(tool) : NULL;
    g_results[g_rn].args        = args ? strdup(args) : NULL;
    g_results[g_rn].first_chunk = (int) g_n;
    g_results[g_rn].n_chunks    = nch;
    g_rn++;

    for (int i = 0; i < nch; i++) {
        g_chunks[g_n].text      = pieces[i];        /* ownership moves to the store */
        g_chunks[g_n].result_id = rid;
        if (vecs) {
            float *v = malloc((size_t) g_dim * sizeof(float));
            if (v) { memcpy(v, vecs + (size_t) i * g_dim, (size_t) g_dim * sizeof(float));
                     g_chunks[g_n].vec = v; }
        }
        g_n++;
    }
    free(pieces);
    free(vecs);
    if (out_result_id) *out_result_id = rid;
    return nch;
}

int toolidx_retrieve(const char *query, int k, float threshold,
                     const char **out_texts, int *out_ids, float *out_scores) {
    if (!query || k <= 0 || g_n == 0 || g_dim <= 0) return 0;
    if (!embed_available()) return 0;

    float *q = malloc((size_t) g_dim * sizeof(float));
    if (!q) return 0;
    if (embed_text(query, q) != 0) { free(q); return 0; }

    /* Vectors are L2-normalized by embed_text/embed_texts, so dot == cosine. */
    float *scores = malloc(g_n * sizeof(float));
    if (!scores) { free(q); return 0; }
    for (size_t i = 0; i < g_n; i++) {
        if (!g_chunks[i].vec) { scores[i] = -1e9f; continue; }
        float s = 0.0f;
        for (int d = 0; d < g_dim; d++) s += q[d] * g_chunks[i].vec[d];
        scores[i] = s;
    }

    int found = 0;
    for (int slot = 0; slot < k; slot++) {          /* repeated max-selection, k is small */
        int best = -1;
        for (size_t i = 0; i < g_n; i++)
            if (scores[i] >= threshold && (best < 0 || scores[i] > scores[best])) best = (int) i;
        if (best < 0) break;
        if (out_texts)  out_texts[found]  = g_chunks[best].text;
        if (out_ids)    out_ids[found]    = best;
        if (out_scores) out_scores[found] = scores[best];
        scores[best] = -1e9f;
        found++;
    }
    free(scores);
    free(q);
    return found;
}

const char *toolidx_get(int chunk_id) {
    if (chunk_id < 0 || (size_t) chunk_id >= g_n) return NULL;
    return g_chunks[chunk_id].text;
}

int toolidx_result_span(int result_id, int *out_first_id, int *out_last_id) {
    if (result_id < 0 || (size_t) result_id >= g_rn) return 0;
    if (out_first_id) *out_first_id = g_results[result_id].first_chunk;
    if (out_last_id)  *out_last_id  = g_results[result_id].first_chunk +
                                      g_results[result_id].n_chunks - 1;
    return g_results[result_id].n_chunks;
}

size_t toolidx_count(void) { return g_n; }

/* Self-test (BASI_TOOLIDX_SELFTEST=1). The chunker is unit-tested without a
 * model; what needs a live embedder is the only question that matters for the
 * whole retrieve-don't-stuff idea: when a needle is buried in a large tool
 * result, does retrieval actually surface it? If it doesn't, the layer trades a
 * guaranteed cost for a silent quality loss. */
void toolidx_selftest(void) {
    fprintf(stderr, "\n=== toolidx self-test ===\n");
    toolidx_clear();

    /* Build a haystack of plausible-but-irrelevant tool output with one needle
       buried in the middle — the shape of a fetched page holding one useful fact. */
    size_t cap = 200000;
    char *doc = malloc(cap);
    if (!doc) return;
    size_t off = 0;
    for (int i = 0; i < 400; i++)
        off += (size_t) snprintf(doc + off, cap - off,
            "src/render_%d.c:%d: static void draw_sprite_%d(struct canvas *c, int x, int y) { blit(c, x, y); }\n",
            i, i * 7 + 3, i);
    size_t needle_at = off;
    off += (size_t) snprintf(doc + off, cap - off,
        "\nThe DPAS instruction on Intel Xe2 requires a subgroup size of sixteen, and the\n"
        "repeat count RC ranges from one to eight, controlling how many output rows a\n"
        "single subgroup produces per systolic call.\n\n");
    for (int i = 400; i < 800; i++)
        off += (size_t) snprintf(doc + off, cap - off,
            "src/audio_%d.c:%d: static int mix_channel_%d(float *buf, int n) { return n; }\n",
            i, i * 5 + 1, i);

    int rid = 0;
    int nch = toolidx_add("web_fetch", "{\"url\":\"https://example/xe2\"}", doc, &rid);
    fprintf(stderr, "[toolidx] %zu bytes -> %d chunks (result %d)\n", off, nch, rid);
    if (nch <= 0) { fprintf(stderr, "[toolidx] indexing FAILED\n"); free(doc); return; }

    /* Which chunk holds the needle? */
    int needle_chunk = -1;
    for (int i = 0; i < (int) g_n; i++)
        if (strstr(g_chunks[i].text, "DPAS instruction")) { needle_chunk = i; break; }
    fprintf(stderr, "[toolidx] needle lives in chunk %d (byte ~%zu of %zu)\n",
            needle_chunk, needle_at, off);

    const char *hits[5]; int ids[5]; float sc[5];
    int got = toolidx_retrieve("What subgroup size does DPAS require on Xe2?",
                               5, 0.0f, hits, ids, sc);
    fprintf(stderr, "[toolidx] retrieved %d chunk(s):\n", got);
    int rank = -1;
    for (int i = 0; i < got; i++) {
        if (ids[i] == needle_chunk) rank = i;
        fprintf(stderr, "   %s [%d] score %.4f  %.62s…\n",
                ids[i] == needle_chunk ? "->" : "  ", ids[i], sc[i], hits[i]);
    }
    if (rank == 0)
        fprintf(stderr, "[toolidx] needle ranked #1 of %d chunks — RETRIEVAL WORKS ✓\n", nch);
    else if (rank > 0)
        fprintf(stderr, "[toolidx] needle ranked #%d (in top-%d) — usable ✓\n", rank + 1, got);
    else
        fprintf(stderr, "[toolidx] needle NOT in top-%d — RETRIEVAL FAILED ✗ "
                        "(the layer would silently lose this fact)\n", got);

    /* Token economics: what the model would see instead of the whole result. */
    size_t inject = 0;
    for (int i = 0; i < got; i++) inject += strlen(hits[i]);
    fprintf(stderr, "[toolidx] would inject %zu B instead of %zu B (%.1fx less)\n",
            inject, off, inject ? (double) off / inject : 0.0);

    free(doc);
    toolidx_clear();
    embed_shutdown();
    fprintf(stderr, "[toolidx] done.\n");
}

void toolidx_clear(void) {
    for (size_t i = 0; i < g_n; i++) { free(g_chunks[i].text); free(g_chunks[i].vec); }
    free(g_chunks); g_chunks = NULL; g_n = g_cap = 0;
    for (size_t i = 0; i < g_rn; i++) { free(g_results[i].tool); free(g_results[i].args); }
    free(g_results); g_results = NULL; g_rn = g_rcap = 0;
}
