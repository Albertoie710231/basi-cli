#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <time.h>       /* clock_gettime — batch self-test timing */

#include <sys/types.h>

#include "util.h"
#include "globals.h"
#include "kb.h"
#include "embed.h"
#include "srvgen.h"    /* spawn/kill the embedder llama-server */
#include "srvchat.h"   /* srvchat_embed — the /embedding HTTP client */

/* ── Static state ──────────────────────────────────────────────────── *
 * The retrieval embedder runs as its OWN spawned llama-server (--embedding) on a
 * dedicated port; embed_text() POSTs to its /embedding endpoint. No in-process
 * model — BASI links no libllama. */

static pid_t  embed_pid     = 0;      /* the embedder llama-server, or 0 */
static int    embed_port    = 8183;   /* separate from the main chat server (8181) */
static int    embed_dim_v   = -1;
static int    embed_n_ctx   = 8192;
static char   embed_err[512] = "";

/* ── Helpers ───────────────────────────────────────────────────────── */

static void set_err(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(embed_err, sizeof(embed_err), fmt, ap);
    va_end(ap);
}

const char *embed_last_error(void) { return embed_err[0] ? embed_err : "ok"; }

/* Pooling passed to the embedder server's --pooling flag. Default "last" (Jina v5
   / Qwen3); override with BASI_EMBED_POOLING=mean|cls|last. */
static const char *pooling_from_env(void) {
    const char *p = getenv("BASI_EMBED_POOLING");
    if (p && (strcmp(p, "mean") == 0 || strcmp(p, "cls") == 0 || strcmp(p, "last") == 0))
        return p;
    return "last";
}

/* Score a model/repo/file name for "embedding-model-ness". Returns -1 if it
 * doesn't look like an embedder at all, else a preference score (higher = more
 * likely a good retrieval embedder). Matches on the NAME, so it works for both a
 * flat file (jina-embeddings-...gguf) and an HF repo dir (models--jinaai--jina-
 * embeddings-v5-...-retrieval-GGUF) whose inner .gguf filename may lack "embed". */
static int embed_name_score(const char *name) {
    char low[1024];
    size_t n = strlen(name), cn = (n < sizeof(low) - 1) ? n : sizeof(low) - 1;
    for (size_t i = 0; i < cn; i++) {
        char c = name[i];
        low[i] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c;
    }
    low[cn] = '\0';
    bool embedish = strstr(low, "embed") || strstr(low, "jina") || strstr(low, "bge") ||
                    strstr(low, "gte")   || strstr(low, "nomic") || strstr(low, "minilm");
    if (!embedish) return -1;
    int s = 0;
    if (strstr(low, "jina"))      s += 4;
    if (strstr(low, "embed"))     s += 3;
    if (strstr(low, "retrieval")) s += 2;
    if (strstr(low, "bge") || strstr(low, "gte") || strstr(low, "nomic")) s += 2;
    if (strstr(low, "qwen3"))     s += 2;
    if (strstr(low, "v5"))        s += 1;
    if (strstr(low, "gemma"))     s += 1;
    return s;
}

/* Find a .gguf inside an HF repo dir's snapshots/<hash>/ (preferring Q4_K_M).
 * Returns malloc'd absolute path or NULL. */
static char *find_gguf_in_snapshots(const char *repo_dir) {
    char snap[1400];
    if ((size_t)snprintf(snap, sizeof(snap), "%s/snapshots", repo_dir) >= sizeof(snap)) return NULL;
    DIR *sd = opendir(snap);
    if (!sd) return NULL;
    char *found = NULL;
    bool found_q4 = false;
    struct dirent *se;
    while ((se = readdir(sd)) != NULL && !found_q4) {
        if (se->d_name[0] == '.') continue;
        char hashdir[1700];
        if ((size_t)snprintf(hashdir, sizeof(hashdir), "%s/%s", snap, se->d_name) >= sizeof(hashdir)) continue;
        DIR *hd = opendir(hashdir);
        if (!hd) continue;
        struct dirent *fe;
        while ((fe = readdir(hd)) != NULL) {
            size_t fn = strlen(fe->d_name);
            if (fn < 5 || strcmp(fe->d_name + fn - 5, ".gguf") != 0) continue;
            char full[2000];
            if ((size_t)snprintf(full, sizeof(full), "%s/%s", hashdir, fe->d_name) >= sizeof(full)) continue;
            bool q4 = strstr(fe->d_name, "Q4_K_M") || strstr(fe->d_name, "q4_k_m");
            if (q4) { free(found); found = strdup(full); found_q4 = true; break; }
            if (!found) found = strdup(full);
        }
        closedir(hd);
    }
    closedir(sd);
    return found;
}

/* Scan a flat dir of .gguf files; update the best/best_score pair for embed-looking ones. */
static void scan_flat_dir(const char *dir, char **best, int *best_score) {
    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        if (de->d_name[0] == '.') continue;
        size_t fn = strlen(de->d_name);
        if (fn < 5 || strcmp(de->d_name + fn - 5, ".gguf") != 0) continue;
        int score = embed_name_score(de->d_name);
        if (score < 0 || score <= *best_score) continue;
        char full[1280];
        if ((size_t)snprintf(full, sizeof(full), "%s/%s", dir, de->d_name) >= sizeof(full)) continue;
        free(*best); *best = strdup(full); *best_score = score;
    }
    closedir(d);
}

/* Path resolution: BASI_EMBED_MODEL override, then ~/.cache/llama.cpp (flat),
 * then the HF hub cache under ~/.cache/huggingface/hub (a models-- repo dir, then
 * its snapshots subtree). Returns malloc'd path or NULL (with err set). */
static char *resolve_embed_model_path(void) {
    const char *override = getenv("BASI_EMBED_MODEL");
    if (override && *override) {
        struct stat st;
        if (stat(override, &st) != 0 || !S_ISREG(st.st_mode)) {
            set_err("BASI_EMBED_MODEL=%s is not a readable file", override);
            return NULL;
        }
        return strdup(override);
    }

    const char *home = getenv("HOME");
    if (!home || !*home) {
        set_err("HOME is not set; cannot locate embedding model. Set BASI_EMBED_MODEL.");
        return NULL;
    }

    char *best = NULL;
    int   best_score = -1;

    char dir[1024];
    if ((size_t)snprintf(dir, sizeof(dir), "%s/.cache/llama.cpp", home) < sizeof(dir))
        scan_flat_dir(dir, &best, &best_score);

    char hub[1024];
    if ((size_t)snprintf(hub, sizeof(hub), "%s/.cache/huggingface/hub", home) < sizeof(hub)) {
        DIR *hd = opendir(hub);
        if (hd) {
            struct dirent *re;
            while ((re = readdir(hd)) != NULL) {
                if (strncmp(re->d_name, "models--", 8) != 0) continue;
                int score = embed_name_score(re->d_name);
                if (score < 0 || score <= best_score) continue;
                char repodir[1300];
                if ((size_t)snprintf(repodir, sizeof(repodir), "%s/%s", hub, re->d_name) >= sizeof(repodir)) continue;
                char *gguf = find_gguf_in_snapshots(repodir);
                if (gguf) { free(best); best = gguf; best_score = score; }
            }
            closedir(hd);
        }
    }

    if (!best)
        set_err("no embedding model found in ~/.cache/llama.cpp or ~/.cache/huggingface/hub. "
                "Set BASI_EMBED_MODEL=<path.gguf>, or download e.g. "
                "jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:Q4_K_M.");
    return best;
}

bool embed_available(void) {
    char *p = resolve_embed_model_path();
    if (p) { free(p); return true; }
    return false;   /* embed_last_error() set by resolve_embed_model_path */
}

/* L2-normalize `v` of length `n` in place. */
static void l2_normalize(float *v, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)v[i] * v[i];
    if (sum <= 0.0) return;
    float inv = (float)(1.0 / sqrt(sum));
    for (int i = 0; i < n; i++) v[i] *= inv;
}

/* ── Public: init / shutdown / dim / embed_text ────────────────────── */

int embed_init(void) {
    if (embed_pid > 0) return 0;   /* embedder server already up */

    char *path = resolve_embed_model_path();
    if (!path) return -1;          /* set_err done by resolver */

    const char *sbin = getenv("BASI_SERVER_BIN");
    if (!sbin || !*sbin) sbin = "/home/alberto/llama.cpp/build_vulkan/bin/llama-server";

    char extra[128];
    snprintf(extra, sizeof extra, "--embedding --pooling %s", pooling_from_env());

    embed_pid = srvgen_spawn(sbin, path, 99 /*small model → all on GPU*/, embed_n_ctx,
                             extra, embed_port, "/tmp/basi_embed_srv.log", 180);
    free(path);
    if (embed_pid < 0) {
        embed_pid = 0;
        set_err("failed to spawn embedder llama-server (see /tmp/basi_embed_srv.log)");
        return -1;
    }

    /* Probe once to learn the embedding dimension (the RAG/reuse vector width). */
    static float probe[8192];
    int d = srvchat_embed(embed_port, "probe", probe, (int)(sizeof probe / sizeof *probe));
    if (d <= 0) {
        srvgen_kill(embed_pid); embed_pid = 0;
        set_err("embedder /embedding probe failed");
        return -1;
    }
    embed_dim_v = d;
    return 0;
}

int embed_dim(void) { return embed_dim_v; }

void embed_shutdown(void) {
    if (embed_pid > 0) { srvgen_kill(embed_pid); embed_pid = 0; }
    embed_dim_v = -1;
}

int embed_text(const char *text, float *out) {
    if (embed_pid <= 0) {
        set_err("embed_text called before embed_init succeeded");
        return -1;
    }
    if (!text || !out) {
        set_err("embed_text: NULL argument");
        return -1;
    }
    if (text[0] == '\0') {                      /* empty text → zero vector */
        for (int i = 0; i < embed_dim_v; i++) out[i] = 0.0f;
        return 0;
    }

    int d = srvchat_embed(embed_port, text, out, embed_dim_v);
    if (d <= 0) {
        set_err("embedder /embedding request failed");
        return -1;
    }
    for (int i = d; i < embed_dim_v; i++) out[i] = 0.0f;   /* pad if short (shouldn't happen) */
    /* The server L2-normalizes (--embd-normalize 2 default); re-normalize for an
       exact unit vector regardless (idempotent) and to match the old contract. */
    l2_normalize(out, embed_dim_v);
    return 0;
}

int embed_texts(const char **texts, int n, float *out) {
    if (embed_pid <= 0) { set_err("embed_texts called before embed_init succeeded"); return -1; }
    if (!texts || !out || n <= 0) { set_err("embed_texts: bad argument"); return -1; }

    int d = srvchat_embed_batch(embed_port, texts, n, out, embed_dim_v);
    if (d <= 0) { set_err("embedder batch /embedding request failed"); return -1; }

    /* Same post-processing as embed_text, per row: pad short, then normalize so
       dot product == cosine. Callers must be able to mix batch and single results
       in one store, so the contract has to be byte-for-byte the same. */
    for (int i = 0; i < n; i++) {
        float *v = out + (size_t) i * (size_t) embed_dim_v;
        for (int k = d; k < embed_dim_v; k++) v[k] = 0.0f;
        l2_normalize(v, embed_dim_v);
    }
    return 0;
}

/* Self-test (BASI_EMBED_BATCH_SELFTEST=1): prove the batch path returns vectors
   IDENTICAL to the single path — a silent divergence would corrupt retrieval
   without ever erroring — and report the speedup on realistically sized chunks. */
void embed_batch_selftest(void) {
    fprintf(stderr, "\n=== embed batch self-test ===\n");
    if (embed_init() != 0) {
        fprintf(stderr, "[embed] init FAILED: %s\n", embed_last_error());
        return;
    }
    const int dim = embed_dim();
    /* Batch size drives the speedup (per-request overhead is amortized), so make
       it sweepable — the right default for the tool-result indexer depends on it. */
    int N = 32;
    { const char *e = getenv("BASI_EMBED_BATCH_N"); if (e && *e && atoi(e) > 0) N = atoi(e); }
    fprintf(stderr, "[embed] ready, dim=%d, n=%d\n", dim, N);

    /* ~200 chars each — matches memory.c's MEM_CHUNK_MAX. Short strings hide the
       per-request overhead entirely and make batching look useless. */
    char **texts = (char **) calloc((size_t) N, sizeof *texts);
    for (int i = 0; i < N; i++) {
        char buf[512];
        snprintf(buf, sizeof buf,
                 "chunk %d: the DPAS instruction performs D = C + A x B with systolic depth "
                 "eight on Intel Xe2 hardware, and the repeat count controls how many output "
                 "rows a single subgroup produces per call.", i);
        texts[i] = strdup(buf);
    }

    float *single = (float *) calloc((size_t) N * dim, sizeof(float));
    float *batch  = (float *) calloc((size_t) N * dim, sizeof(float));
    if (!texts || !single || !batch) { fprintf(stderr, "[embed] OOM\n"); return; }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int bad = 0;
    for (int i = 0; i < N; i++)
        if (embed_text(texts[i], single + (size_t) i * dim) != 0) bad++;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double s_one = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    int rc = embed_texts((const char **) texts, N, batch);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double s_bat = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    if (rc != 0) { fprintf(stderr, "[embed] BATCH FAILED: %s\n", embed_last_error()); return; }

    /* Cosine between the two versions of each row must be ~1.0. */
    double worst = 1.0;
    int    worst_i = -1;
    for (int i = 0; i < N; i++) {
        const float *a = single + (size_t) i * dim, *b = batch + (size_t) i * dim;
        double dot = 0.0;
        for (int k = 0; k < dim; k++) dot += (double) a[k] * (double) b[k];
        if (dot < worst) { worst = dot; worst_i = i; }
    }

    fprintf(stderr, "[embed] single: %.2fs (%.1f ms/text, %d failed)\n",
            s_one, 1000.0 * s_one / N, bad);
    fprintf(stderr, "[embed] batch : %.2fs (%.1f ms/text)\n", s_bat, 1000.0 * s_bat / N);
    fprintf(stderr, "[embed] speedup: %.1fx\n", s_bat > 0 ? s_one / s_bat : 0.0);
    fprintf(stderr, "[embed] worst cosine(single,batch) = %.6f (row %d) -> %s\n",
            worst, worst_i, worst > 0.9999 ? "IDENTICAL ✓" : "DIVERGENT ✗");

    /* Distinct texts must still yield distinct vectors — catches a batch path that
       silently returns the same embedding N times. */
    double cross = 0.0;
    for (int k = 0; k < dim; k++) cross += (double) batch[k] * (double) batch[(size_t) dim + k];
    fprintf(stderr, "[embed] cosine(row0,row1) = %.4f -> %s\n",
            cross, cross < 0.999 ? "distinct ✓" : "SUSPICIOUS (rows identical)");

    for (int i = 0; i < N; i++) free(texts[i]);
    free(texts); free(single); free(batch);
    embed_shutdown();
    fprintf(stderr, "[embed] done.\n");
}

/* ── Chunker ───────────────────────────────────────────────────────── *
 *
 * One chunk per H2 section (with nested H3s included). Content before the
 * first H2 becomes the "_intro" chunk. Each chunk capped at CHUNK_CAP
 * bytes (truncated, not split — chunks above the cap are rare given how
 * we author docs). Anchor uses the heading slug; "_intro" for pre-H2.
 */

#define CHUNK_CAP 4096

typedef struct {
    char  *path;     /* full path, malloc'd, e.g. ".basi/knowledge/notes/x.md" */
    char  *anchor;   /* heading text (utf-8) or "_intro" */
    char  *text;     /* malloc'd, NUL-terminated chunk body (capped) */
    long   mtime;    /* source file mtime when chunked */
} Chunk;

static void chunk_free(Chunk *c) {
    free(c->path); free(c->anchor); free(c->text);
    c->path = c->anchor = c->text = NULL;
}

/* Trim ASCII whitespace at both ends of a slice. */
static void slice_trim(const char **s, size_t *len) {
    while (*len && (**s == ' ' || **s == '\t' || **s == '\r' || **s == '\n')) {
        (*s)++; (*len)--;
    }
    while (*len && (((*s)[*len - 1] == ' ') || ((*s)[*len - 1] == '\t') ||
                    ((*s)[*len - 1] == '\r') || ((*s)[*len - 1] == '\n'))) {
        (*len)--;
    }
}

static char *slice_dup(const char *s, size_t len) {
    char *out = malloc(len + 1);
    memcpy(out, s, len);
    out[len] = '\0';
    return out;
}

/* Heading line? Returns level (1..6) if so and writes the heading text
 * slice via out_text/out_len; 0 otherwise. */
static int heading_line(const char *p, size_t len,
                        const char **out_text, size_t *out_len) {
    if (len == 0 || *p != '#') return 0;
    int level = 0;
    while (level < 6 && (size_t)level < len && p[level] == '#') level++;
    if (level == 0 || (size_t)level >= len || p[level] != ' ') return 0;
    const char *h = p + level + 1;
    size_t hlen = len - level - 1;
    while (hlen && (*h == ' ' || *h == '\t')) { h++; hlen--; }
    while (hlen && (h[hlen - 1] == ' ' || h[hlen - 1] == '\t' ||
                    h[hlen - 1] == '#' || h[hlen - 1] == '\r')) hlen--;
    *out_text = h;
    *out_len = hlen;
    return level;
}

/* Append chunk to a dynamic array, growing as needed. */
static void chunks_push(Chunk **arr, size_t *n, size_t *cap,
                        const char *path, const char *anchor, size_t alen,
                        const char *text, size_t tlen, long mtime) {
    if (*n == *cap) {
        *cap = *cap ? *cap * 2 : 16;
        *arr = realloc(*arr, *cap * sizeof(Chunk));
    }
    Chunk *c = &(*arr)[*n];
    c->path   = strdup(path);
    c->anchor = slice_dup(anchor, alen);
    if (tlen > CHUNK_CAP) tlen = CHUNK_CAP;
    c->text   = slice_dup(text, tlen);
    c->mtime  = mtime;
    (*n)++;
}

/* Chunk one markdown file. Caller frees the array via chunks_free. */
static int chunk_file(const char *path, long mtime,
                      Chunk **out_arr, size_t *out_n) {
    *out_arr = NULL; *out_n = 0;
    size_t fl = 0;
    char *buf = kb_read_file(path, &fl);
    if (!buf) return -1;
    KbFrontmatter fm;
    kb_parse_frontmatter(buf, fl, &fm);
    const char *body = buf + fm.body_offset;
    size_t body_len = fl - fm.body_offset;
    kb_fm_free(&fm);

    Chunk *arr = NULL;
    size_t n = 0, cap = 0;

    /* Walk lines. Track current H2 boundaries. */
    const char *cur_anchor = "_intro";
    size_t cur_anchor_len = 6;
    const char *section_start = body;
    const char *p = body;
    const char *end = body + body_len;
    while (p < end) {
        const char *le = p;
        while (le < end && *le != '\n') le++;
        const char *htext;
        size_t hlen;
        int level = heading_line(p, le - p, &htext, &hlen);
        if (level == 2) {
            /* close previous section */
            const char *s = section_start;
            size_t slen = p - section_start;
            slice_trim(&s, &slen);
            if (slen > 0) {
                chunks_push(&arr, &n, &cap, path,
                    cur_anchor, cur_anchor_len, s, slen, mtime);
            }
            cur_anchor     = htext;
            cur_anchor_len = hlen;
            section_start  = p;     /* keep heading line in chunk text */
        }
        p = (le < end) ? le + 1 : end;
    }
    /* Tail section. */
    const char *s = section_start;
    size_t slen = end - section_start;
    slice_trim(&s, &slen);
    if (slen > 0) {
        chunks_push(&arr, &n, &cap, path,
            cur_anchor, cur_anchor_len, s, slen, mtime);
    }

    free(buf);
    *out_arr = arr;
    *out_n = n;
    return 0;
}

static void chunks_free(Chunk *arr, size_t n) {
    for (size_t i = 0; i < n; i++) chunk_free(&arr[i]);
    free(arr);
}

/* ── Vector store (.basi/knowledge/.embeddings.bin) ────────────────── *
 *
 * On-disk layout (little-endian, host architecture):
 *   uint32 magic   = 'BASE'  (0x45534142)
 *   uint32 version = 1
 *   uint32 dim
 *   uint32 n_entries
 *   for each entry:
 *     uint16 path_len   ; char[path_len]
 *     uint16 anchor_len ; char[anchor_len]
 *     uint32 text_len   ; char[text_len]   (NOT NUL-terminated on disk)
 *     int64  mtime
 *     float[dim]
 */

#define EMB_FILE  ".basi/knowledge/.embeddings.bin"
#define EMB_MAGIC 0x45534142u  /* 'BASE' little-endian */
#define EMB_VER   1u

typedef struct {
    char  *path;
    char  *anchor;
    char  *text;
    long   mtime;
    float *vec;       /* malloc'd, length = store->dim */
} VecEntry;

typedef struct {
    int       dim;
    VecEntry *e;
    size_t    n;
    size_t    cap;
} VecStore;

static void vs_init(VecStore *s, int dim) {
    s->dim = dim; s->e = NULL; s->n = 0; s->cap = 0;
}

static void vs_clear(VecStore *s) {
    for (size_t i = 0; i < s->n; i++) {
        free(s->e[i].path); free(s->e[i].anchor); free(s->e[i].text);
        free(s->e[i].vec);
    }
    free(s->e);
    s->e = NULL; s->n = 0; s->cap = 0;
}

static void vs_push(VecStore *s, const char *path, const char *anchor,
                    const char *text, long mtime, float *vec /* takes ownership */) {
    if (s->n == s->cap) {
        s->cap = s->cap ? s->cap * 2 : 32;
        s->e = realloc(s->e, s->cap * sizeof(VecEntry));
    }
    VecEntry *v = &s->e[s->n++];
    v->path   = strdup(path);
    v->anchor = strdup(anchor);
    v->text   = strdup(text);
    v->mtime  = mtime;
    v->vec    = vec;
}

/* Returns 0 on success (or no file present); -1 on read/format error. */
static int vs_load(VecStore *s) {
    FILE *f = fopen(EMB_FILE, "rb");
    if (!f) {
        if (errno == ENOENT) return 0;
        return -1;
    }
    uint32_t magic, ver, dim, count;
    if (fread(&magic, 4, 1, f) != 1 || fread(&ver, 4, 1, f) != 1 ||
        fread(&dim, 4, 1, f) != 1   || fread(&count, 4, 1, f) != 1) {
        fclose(f); return -1;
    }
    if (magic != EMB_MAGIC || ver != EMB_VER) { fclose(f); return -1; }
    if ((int)dim != s->dim) {
        /* Different model dim → invalidate the store. */
        fclose(f); return -1;
    }
    for (uint32_t i = 0; i < count; i++) {
        uint16_t plen, alen; uint32_t tlen; int64_t mtime;
        if (fread(&plen, 2, 1, f) != 1) { fclose(f); return -1; }
        char *path = malloc(plen + 1);
        if (fread(path, 1, plen, f) != plen) { free(path); fclose(f); return -1; }
        path[plen] = '\0';
        if (fread(&alen, 2, 1, f) != 1) { free(path); fclose(f); return -1; }
        char *anc = malloc(alen + 1);
        if (fread(anc, 1, alen, f) != alen) { free(path); free(anc); fclose(f); return -1; }
        anc[alen] = '\0';
        if (fread(&tlen, 4, 1, f) != 1) { free(path); free(anc); fclose(f); return -1; }
        char *text = malloc(tlen + 1);
        if (fread(text, 1, tlen, f) != tlen) { free(path); free(anc); free(text); fclose(f); return -1; }
        text[tlen] = '\0';
        if (fread(&mtime, 8, 1, f) != 1) { free(path); free(anc); free(text); fclose(f); return -1; }
        float *vec = malloc(s->dim * sizeof(float));
        if (fread(vec, sizeof(float), s->dim, f) != (size_t)s->dim) {
            free(path); free(anc); free(text); free(vec); fclose(f); return -1;
        }
        /* push raw — don't strdup again (transfer ownership) */
        if (s->n == s->cap) {
            s->cap = s->cap ? s->cap * 2 : 32;
            s->e = realloc(s->e, s->cap * sizeof(VecEntry));
        }
        s->e[s->n++] = (VecEntry){ path, anc, text, (long)mtime, vec };
    }
    fclose(f);
    return 0;
}

static int vs_save(const VecStore *s) {
    if (kb_ensure_dirs() != 0) return -1;
    char tmp[256];
    snprintf(tmp, sizeof(tmp), "%s.tmp", EMB_FILE);
    FILE *f = fopen(tmp, "wb");
    if (!f) return -1;
    uint32_t magic = EMB_MAGIC, ver = EMB_VER;
    uint32_t dim = (uint32_t)s->dim, count = (uint32_t)s->n;
    if (fwrite(&magic, 4, 1, f) != 1 || fwrite(&ver, 4, 1, f) != 1 ||
        fwrite(&dim, 4, 1, f) != 1   || fwrite(&count, 4, 1, f) != 1) {
        fclose(f); unlink(tmp); return -1;
    }
    for (size_t i = 0; i < s->n; i++) {
        const VecEntry *v = &s->e[i];
        uint16_t plen = (uint16_t)strlen(v->path);
        uint16_t alen = (uint16_t)strlen(v->anchor);
        uint32_t tlen = (uint32_t)strlen(v->text);
        int64_t  mt   = (int64_t)v->mtime;
        if (fwrite(&plen, 2, 1, f) != 1 || fwrite(v->path, 1, plen, f) != plen ||
            fwrite(&alen, 2, 1, f) != 1 || fwrite(v->anchor, 1, alen, f) != alen ||
            fwrite(&tlen, 4, 1, f) != 1 || fwrite(v->text, 1, tlen, f) != tlen ||
            fwrite(&mt, 8, 1, f) != 1   ||
            fwrite(v->vec, sizeof(float), s->dim, f) != (size_t)s->dim) {
            fclose(f); unlink(tmp); return -1;
        }
    }
    if (fclose(f) != 0) { unlink(tmp); return -1; }
    if (rename(tmp, EMB_FILE) != 0) { unlink(tmp); return -1; }
    return 0;
}

/* Drop entries whose source file is gone or whose recorded mtime is stale.
 * Files staying are returned via `keep_seen` keyed by (path, anchor). */
typedef struct { const char *path; const char *anchor; } SeenKey;

static int seen_has(const SeenKey *seen, size_t n,
                    const char *path, const char *anchor) {
    for (size_t i = 0; i < n; i++) {
        if (strcmp(seen[i].path, path) == 0 &&
            strcmp(seen[i].anchor, anchor) == 0) return 1;
    }
    return 0;
}

/* Walk-the-KB ctx for kb_walk_dir. */
typedef struct {
    Chunk *all;
    size_t n, cap;
} ChunkCollect;

static int collect_visit(const char *path, void *ud) {
    ChunkCollect *cc = (ChunkCollect *)ud;
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    /* Skip the embeddings sidecar itself defensively. */
    if (strstr(path, ".embeddings.bin")) return 0;
    Chunk *cs = NULL; size_t nc = 0;
    if (chunk_file(path, (long)st.st_mtime, &cs, &nc) != 0) return 0;
    for (size_t i = 0; i < nc; i++) {
        if (cc->n == cc->cap) {
            cc->cap = cc->cap ? cc->cap * 2 : 32;
            cc->all = realloc(cc->all, cc->cap * sizeof(Chunk));
        }
        cc->all[cc->n++] = cs[i];   /* transfer ownership */
    }
    free(cs);
    return 0;
}

/* Reconcile store with current on-disk KB. Embeds anything new or stale.
 * Returns number of (re)embedded chunks, or -1 on error. */
static int vs_sync(VecStore *s, int *out_kept, int *out_dropped) {
    *out_kept = 0; *out_dropped = 0;

    /* 1. Collect every current chunk. */
    ChunkCollect cc = { NULL, 0, 0 };
    kb_walk_dir(KB_NOTES_DIR,  collect_visit, &cc);
    kb_walk_dir(KB_PINNED_DIR, collect_visit, &cc);
    kb_walk_dir(KB_DOCS_DIR,   collect_visit, &cc);

    /* 2. Build a (path, anchor) → mtime index of current chunks. */
    /* 3. Walk store: drop entries whose current chunk is missing or whose
     *    mtime is stale. */
    VecStore kept;
    vs_init(&kept, s->dim);
    SeenKey *seen = malloc(s->n * sizeof(SeenKey) + sizeof(SeenKey));
    size_t  n_seen = 0;
    for (size_t i = 0; i < s->n; i++) {
        VecEntry *v = &s->e[i];
        long want_mtime = -1;
        for (size_t j = 0; j < cc.n; j++) {
            if (strcmp(cc.all[j].path, v->path) == 0 &&
                strcmp(cc.all[j].anchor, v->anchor) == 0) {
                want_mtime = cc.all[j].mtime;
                break;
            }
        }
        if (want_mtime != -1 && want_mtime == v->mtime) {
            /* Keep — transfer ownership. */
            seen[n_seen++] = (SeenKey){ v->path, v->anchor };
            if (kept.n == kept.cap) {
                kept.cap = kept.cap ? kept.cap * 2 : 32;
                kept.e = realloc(kept.e, kept.cap * sizeof(VecEntry));
            }
            kept.e[kept.n++] = *v;
            v->path = v->anchor = v->text = NULL;
            v->vec = NULL;
            (*out_kept)++;
        } else {
            (*out_dropped)++;
        }
    }
    /* Free anything that wasn't kept. */
    for (size_t i = 0; i < s->n; i++) {
        free(s->e[i].path); free(s->e[i].anchor);
        free(s->e[i].text); free(s->e[i].vec);
    }
    free(s->e);
    *s = kept;

    /* 4. Embed every current chunk that isn't in the kept set. */
    int n_embedded = 0;
    for (size_t j = 0; j < cc.n; j++) {
        if (seen_has(seen, n_seen, cc.all[j].path, cc.all[j].anchor)) continue;
        float *vec = malloc(s->dim * sizeof(float));
        if (embed_text(cc.all[j].text, vec) != 0) {
            free(vec);
            chunks_free(cc.all + j, 1);   /* free this single chunk */
            continue;
        }
        vs_push(s, cc.all[j].path, cc.all[j].anchor, cc.all[j].text,
                cc.all[j].mtime, vec);
        n_embedded++;
    }
    free(seen);

    /* Free remaining collected chunks (those we didn't transfer in vs_push,
     * i.e. all of them — vs_push strdup'd everything). */
    chunks_free(cc.all, cc.n);

    return n_embedded;
}

/* ── Top-K search ──────────────────────────────────────────────────── */

typedef struct { int idx; float score; } Hit;

static int hit_cmp_desc(const void *a, const void *b) {
    float da = ((const Hit *)b)->score - ((const Hit *)a)->score;
    return (da > 0) - (da < 0);
}

static void score_topk(const VecStore *s, const float *q,
                       int k, Hit *out, int *out_n) {
    Hit *all = malloc(s->n * sizeof(Hit) + sizeof(Hit));
    for (size_t i = 0; i < s->n; i++) {
        double dot = 0.0;
        const float *v = s->e[i].vec;
        for (int d = 0; d < s->dim; d++) dot += (double)v[d] * q[d];
        all[i].idx = (int)i;
        all[i].score = (float)dot;
    }
    qsort(all, s->n, sizeof(Hit), hit_cmp_desc);
    int n = (int)s->n < k ? (int)s->n : k;
    for (int i = 0; i < n; i++) out[i] = all[i];
    *out_n = n;
    free(all);
}

/* ── Tool entry ────────────────────────────────────────────────────── */

#define VS_TOPK   5
#define VS_SNIPPET_CAP 320

char *execute_docs_vector_search(const char *args) {
    while (*args == ' ' || *args == '\t' || *args == '\n') args++;
    if (!*args) {
        return strdup(
            "docs_vector_search not allowed: missing query. "
            "Example: docs_vector_search how does the planner gate tools");
    }
    /* Trim trailing whitespace. */
    size_t qlen = strlen(args);
    while (qlen && (args[qlen - 1] == ' ' || args[qlen - 1] == '\t' ||
                    args[qlen - 1] == '\n' || args[qlen - 1] == '\r')) qlen--;
    if (qlen == 0) {
        return strdup("docs_vector_search not allowed: empty query.");
    }
    char *query = malloc(qlen + 1);
    memcpy(query, args, qlen);
    query[qlen] = '\0';

    if (embed_init() != 0) {
        char *msg = malloc(1024);
        snprintf(msg, 1024,
            "docs_vector_search failed: %s\n"
            "(use docs_search for literal grep instead, or set BASI_EMBED_MODEL.)",
            embed_last_error());
        free(query);
        return msg;
    }

    VecStore store;
    vs_init(&store, embed_dim());
    vs_load(&store);   /* tolerate missing file */

    int kept = 0, dropped = 0;
    int embedded = vs_sync(&store, &kept, &dropped);
    if (embedded < 0) {
        vs_clear(&store);
        free(query);
        return strdup("docs_vector_search failed during corpus sync.");
    }
    if (embedded > 0 || dropped > 0) vs_save(&store);

    if (store.n == 0) {
        vs_clear(&store);
        free(query);
        return strdup(
            "docs_vector_search: knowledge base is empty. "
            "Add files via 'basi-cli docs add' or '/note ...'.\n");
    }

    /* Embed query and score. */
    float *qvec = malloc(store.dim * sizeof(float));
    if (embed_text(query, qvec) != 0) {
        char *msg = malloc(512);
        snprintf(msg, 512, "docs_vector_search failed embedding query: %.440s",
                 embed_last_error());
        free(qvec); vs_clear(&store); free(query);
        return msg;
    }

    Hit hits[VS_TOPK];
    int n_hits = 0;
    score_topk(&store, qvec, VS_TOPK, hits, &n_hits);
    free(qvec);

    StringBuf out;
    sb_init(&out);
    char header[256];
    snprintf(header, sizeof(header),
        "docs_vector_search: top %d match%s for \"%.80s\" "
        "(corpus: %zu chunks; reembedded %d this call):\n\n",
        n_hits, n_hits == 1 ? "" : "es", query, store.n, embedded);
    sb_append_str(&out, header);

    for (int i = 0; i < n_hits; i++) {
        const VecEntry *v = &store.e[hits[i].idx];
        const char *rel = kb_strip_know_prefix(v->path);
        char line[256];
        snprintf(line, sizeof(line),
            "%2d. %s#%s   score=%.3f\n",
            i + 1, rel, v->anchor, hits[i].score);
        sb_append_str(&out, line);
        /* Snippet: first VS_SNIPPET_CAP chars of chunk text. */
        size_t tlen = strlen(v->text);
        size_t slen = tlen < VS_SNIPPET_CAP ? tlen : VS_SNIPPET_CAP;
        sb_append_str(&out, "    │ ");
        for (size_t k = 0; k < slen; k++) {
            char c = v->text[k];
            if (c == '\n') { sb_append_str(&out, "\n    │ "); continue; }
            sb_append_char(&out, c);
        }
        if (slen < tlen) sb_append_str(&out, " …");
        sb_append_char(&out, '\n');
    }

    vs_clear(&store);
    free(query);
    return sb_to_str(&out);
}
