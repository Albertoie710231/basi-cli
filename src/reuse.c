#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>

#include "util.h"
#include "kb.h"       /* kb_ensure_dirs — the store lives under .basi/ */
#include "embed.h"    /* embed_available / embed_init / embed_dim / embed_text */
#include "reuse.h"

/* ── Tunables ──────────────────────────────────────────────────────────
 * All overridable at runtime so the gate can be calibrated on a benchmark
 * without a rebuild. */
#define SYM_STORE      ".basi/.symbols.bin"
#define SYM_MAGIC      0x4d595342u          /* 'BSYM' little-endian */
#define SYM_VER        1u
#define MIN_BODY       60                    /* ignore trivial one-liners */
#define EMBED_CAP      1600                  /* chars of a func fed to the embedder */
#define MAX_FILE_BYTES (512 * 1024)          /* skip generated / vendored blobs */
#define DEFAULT_TAU    0.80f

static float env_tau(void) {
    const char *e = getenv("BASI_REUSE_TAU");
    if (e && *e) { float v = (float)atof(e); if (v > 0.0f && v <= 1.0f) return v; }
    return DEFAULT_TAU;
}

static bool env_flag(const char *name) {
    const char *e = getenv(name);
    if (!e || !*e) return false;
    char c = *e;
    return c == '1' || c == 'o' || c == 'O' || c == 't' || c == 'T' || c == 'y' || c == 'Y';
}

int reuse_gate_enabled(void) { return env_flag("BASI_REUSE_GATE"); }

/* ── Heuristic C/C++ function extractor ─────────────────────────────────
 * Runs identically on files-on-disk (the index) and on an edit's REPLACE text
 * (the candidate), so the two sides embed comparably. Not a parser: it finds
 * top-level `name(...) { ... }` definitions and skips everything else
 * (prototypes, structs, initializers, macros). See make_skeleton() for how
 * comments/strings are neutralized before the structural scan. */

typedef struct {
    char *name;
    char *sig;    /* signature text up to the '{', whitespace-collapsed */
    char *body;   /* the '{...}' block, verbatim (capped at use sites) */
    int   line;   /* 1-based line of the definition start */
} FuncDef;

static void funcdef_free(FuncDef *f) {
    free(f->name); free(f->sig); free(f->body);
    f->name = f->sig = f->body = NULL;
}

/* Produce a same-length copy of `s` with comment and string/char-literal
 * contents blanked to spaces (newlines preserved), so brace/paren/semicolon
 * counting isn't fooled by punctuation inside them. */
static char *make_skeleton(const char *s, size_t n) {
    char *sk = malloc(n + 1);
    enum { CODE, LC, BC, STR, CH } st = CODE;
    size_t i = 0;
    while (i < n) {
        char c = s[i];
        char d = (i + 1 < n) ? s[i + 1] : '\0';
        if (st == CODE) {
            if (c == '/' && d == '/') { sk[i] = ' '; sk[i + 1] = ' '; i += 2; st = LC;  continue; }
            if (c == '/' && d == '*') { sk[i] = ' '; sk[i + 1] = ' '; i += 2; st = BC;  continue; }
            if (c == '"')  { sk[i] = ' '; i++; st = STR; continue; }
            if (c == '\'') { sk[i] = ' '; i++; st = CH;  continue; }
            sk[i] = c; i++; continue;
        }
        if (st == LC) {
            if (c == '\n') { sk[i] = '\n'; st = CODE; } else sk[i] = ' ';
            i++; continue;
        }
        if (st == BC) {
            if (c == '*' && d == '/') { sk[i] = ' '; sk[i + 1] = ' '; i += 2; st = CODE; continue; }
            sk[i] = (c == '\n') ? '\n' : ' '; i++; continue;
        }
        /* STR or CH: identical handling apart from the closing delimiter. */
        if (c == '\\') { sk[i] = ' '; if (i + 1 < n) sk[i + 1] = ' '; i += 2; continue; }
        if ((st == STR && c == '"') || (st == CH && c == '\'')) { sk[i] = ' '; i++; st = CODE; continue; }
        sk[i] = (c == '\n') ? '\n' : ' '; i++; continue;
    }
    sk[n] = '\0';
    return sk;
}

static bool ident_reject(const char *w) {
    static const char *kw[] = {
        "if", "for", "while", "switch", "return", "sizeof", "do", "else",
        "case", "default", "goto", "typedef", "struct", "union", "enum",
        "static", "const", "void", "unsigned", "signed", "extern", "inline",
        "register", "volatile", "__attribute__", "class", "namespace", "template",
    };
    for (size_t i = 0; i < sizeof(kw) / sizeof(kw[0]); i++)
        if (strcmp(w, kw[i]) == 0) return true;
    return false;
}

static size_t match_brace(const char *sk, size_t len, size_t pos) {
    int depth = 0;
    for (size_t k = pos; k < len; k++) {
        if (sk[k] == '{') depth++;
        else if (sk[k] == '}') { depth--; if (depth == 0) return k; }
    }
    return len;   /* unbalanced */
}

/* Duplicate src[a,b) with runs of whitespace collapsed to single spaces and
 * ends trimmed — a clean one-line signature for display. */
static char *collapse_range(const char *src, size_t a, size_t b) {
    StringBuf sb; sb_init(&sb);
    bool in_ws = true;   /* start true so leading ws is dropped */
    for (size_t i = a; i < b; i++) {
        char c = src[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!in_ws) { sb_append_char(&sb, ' '); in_ws = true; }
        } else { sb_append_char(&sb, c); in_ws = false; }
    }
    /* drop a trailing space if any */
    if (sb.len && sb.data[sb.len - 1] == ' ') sb.len--;
    return sb_to_str(&sb);
}

static char *slice_dup(const char *src, size_t a, size_t b) {
    size_t n = b - a;
    char *o = malloc(n + 1);
    memcpy(o, src + a, n);
    o[n] = '\0';
    return o;
}

static FuncDef *extract_funcs(const char *src, size_t len, int *out_n) {
    *out_n = 0;
    if (!src || len == 0) return NULL;
    char *sk = make_skeleton(src, len);
    FuncDef *arr = NULL; int n = 0, cap = 0;

    size_t i = 0;
    while (i < len) {
        while (i < len && (sk[i] == ' ' || sk[i] == '\t' || sk[i] == '\n' || sk[i] == '\r')) i++;
        if (i >= len) break;

        if (sk[i] == '#') {                     /* preprocessor line (handles \-continuation) */
            while (i < len && sk[i] != '\n') {
                if (sk[i] == '\\' && i + 1 < len && sk[i + 1] == '\n') i++;
                i++;
            }
            continue;
        }
        if (sk[i] == '}' || sk[i] == ';') { i++; continue; }
        if (sk[i] == '{') { size_t e = match_brace(sk, len, i); i = (e < len) ? e + 1 : len; continue; }

        /* Scan to the next top-level '{' or ';', remembering the first '('. */
        size_t start = i, j = i;
        long paren = -1;
        while (j < len && sk[j] != '{' && sk[j] != ';') {
            if (sk[j] == '(' && paren < 0) paren = (long)j;
            j++;
        }
        if (j >= len) break;
        if (sk[j] == ';') { i = j + 1; continue; }   /* declaration / prototype / global */

        size_t bpos = j;
        size_t bend = match_brace(sk, len, bpos);
        bool is_func = (paren >= 0 && (size_t)paren < bpos);

        if (is_func) {
            long p = paren - 1;
            while (p >= (long)start && (sk[p] == ' ' || sk[p] == '\t' || sk[p] == '\n' || sk[p] == '\r')) p--;
            long e = p;
            while (p >= (long)start && (isalnum((unsigned char)sk[p]) || sk[p] == '_')) p--;
            long nstart = p + 1;
            if (e >= nstart) {
                size_t nlen = (size_t)(e - nstart + 1);
                char *name = malloc(nlen + 1);
                memcpy(name, src + nstart, nlen);
                name[nlen] = '\0';
                if ((isalpha((unsigned char)name[0]) || name[0] == '_') && !ident_reject(name)) {
                    if (n >= cap) { cap = cap ? cap * 2 : 8; arr = realloc(arr, cap * sizeof(FuncDef)); }
                    int line = 1;
                    for (size_t z = 0; z < start; z++) if (src[z] == '\n') line++;
                    size_t body_end = (bend < len) ? bend + 1 : len;
                    arr[n].name = name;
                    arr[n].sig  = collapse_range(src, start, bpos);
                    arr[n].body = slice_dup(src, bpos, body_end);
                    arr[n].line = line;
                    n++;
                } else {
                    free(name);
                }
            }
        }
        i = (bend < len) ? bend + 1 : len;
    }

    free(sk);
    *out_n = n;
    return arr;
}

/* Embedding payload: name + signature + (capped) body. Same recipe on both
 * index and candidate sides. */
static char *build_payload(const FuncDef *f) {
    StringBuf sb; sb_init(&sb);
    sb_append_str(&sb, f->name);
    sb_append_char(&sb, '\n');
    sb_append_str(&sb, f->sig);
    sb_append_char(&sb, '\n');
    size_t blen = strlen(f->body);
    if (blen > EMBED_CAP) blen = EMBED_CAP;
    sb_append(&sb, f->body, blen);
    return sb_to_str(&sb);
}

/* Whole-word occurrence of `name` in `hay` (so editing an existing function
 * whose name is in the SEARCH text isn't treated as adding a new one). */
static bool name_in_text(const char *hay, const char *name) {
    if (!hay || !*hay) return false;
    size_t nl = strlen(name);
    const char *p = hay;
    while ((p = strstr(p, name)) != NULL) {
        char before = (p == hay) ? ' ' : p[-1];
        char after  = p[nl];
        bool bok = !(isalnum((unsigned char)before) || before == '_');
        bool aok = !(isalnum((unsigned char)after)  || after == '_');
        if (bok && aok) return true;
        p += nl;
    }
    return false;
}

/* ── Symbol store (.basi/.symbols.bin) ───────────────────────────────────
 * Mirrors embed.c's on-disk vector store, keyed per-function. Persisted so a
 * fresh `basi` launch doesn't re-embed an unchanged tree; invalidated per-file
 * on mtime. */

typedef struct {
    char  *file;
    char  *name;
    char  *sig;
    int    line;
    long   mtime;
    float *vec;      /* length = store->dim */
} SymEntry;

typedef struct {
    int       dim;
    SymEntry *e;
    size_t    n, cap;
} SymStore;

static void sym_init(SymStore *s, int dim) { s->dim = dim; s->e = NULL; s->n = s->cap = 0; }

static void sym_entry_free(SymEntry *v) {
    free(v->file); free(v->name); free(v->sig); free(v->vec);
    v->file = v->name = v->sig = NULL; v->vec = NULL;
}

static void sym_clear(SymStore *s) {
    for (size_t i = 0; i < s->n; i++) sym_entry_free(&s->e[i]);
    free(s->e);
    s->e = NULL; s->n = s->cap = 0;
}

static void sym_push(SymStore *s, char *file, char *name, char *sig,
                     int line, long mtime, float *vec /* all owned */) {
    if (s->n == s->cap) { s->cap = s->cap ? s->cap * 2 : 32; s->e = realloc(s->e, s->cap * sizeof(SymEntry)); }
    SymEntry *v = &s->e[s->n++];
    v->file = file; v->name = name; v->sig = sig;
    v->line = line; v->mtime = mtime; v->vec = vec;
}

static int sym_load(SymStore *s) {
    FILE *f = fopen(SYM_STORE, "rb");
    if (!f) return (errno == ENOENT) ? 0 : -1;
    uint32_t magic, ver, dim, count;
    if (fread(&magic, 4, 1, f) != 1 || fread(&ver, 4, 1, f) != 1 ||
        fread(&dim, 4, 1, f) != 1   || fread(&count, 4, 1, f) != 1) { fclose(f); return -1; }
    if (magic != SYM_MAGIC || ver != SYM_VER || (int)dim != s->dim) { fclose(f); return -1; }
    for (uint32_t i = 0; i < count; i++) {
        uint16_t flen, nlen, slen; int32_t line; int64_t mtime;
        if (fread(&flen, 2, 1, f) != 1) { fclose(f); return -1; }
        char *file = malloc(flen + 1);
        if (fread(file, 1, flen, f) != flen) { free(file); fclose(f); return -1; }
        file[flen] = '\0';
        if (fread(&nlen, 2, 1, f) != 1) { free(file); fclose(f); return -1; }
        char *name = malloc(nlen + 1);
        if (fread(name, 1, nlen, f) != nlen) { free(file); free(name); fclose(f); return -1; }
        name[nlen] = '\0';
        if (fread(&slen, 2, 1, f) != 1) { free(file); free(name); fclose(f); return -1; }
        char *sig = malloc(slen + 1);
        if (fread(sig, 1, slen, f) != slen) { free(file); free(name); free(sig); fclose(f); return -1; }
        sig[slen] = '\0';
        if (fread(&line, 4, 1, f) != 1 || fread(&mtime, 8, 1, f) != 1) {
            free(file); free(name); free(sig); fclose(f); return -1;
        }
        float *vec = malloc((size_t)s->dim * sizeof(float));
        if (fread(vec, sizeof(float), s->dim, f) != (size_t)s->dim) {
            free(file); free(name); free(sig); free(vec); fclose(f); return -1;
        }
        sym_push(s, file, name, sig, (int)line, (long)mtime, vec);
    }
    fclose(f);
    return 0;
}

static int sym_save(const SymStore *s) {
    if (kb_ensure_dirs() != 0) return -1;
    char tmp[256];
    snprintf(tmp, sizeof(tmp), "%s.tmp", SYM_STORE);
    FILE *f = fopen(tmp, "wb");
    if (!f) return -1;
    uint32_t magic = SYM_MAGIC, ver = SYM_VER, dim = (uint32_t)s->dim, count = (uint32_t)s->n;
    if (fwrite(&magic, 4, 1, f) != 1 || fwrite(&ver, 4, 1, f) != 1 ||
        fwrite(&dim, 4, 1, f) != 1   || fwrite(&count, 4, 1, f) != 1) { fclose(f); unlink(tmp); return -1; }
    for (size_t i = 0; i < s->n; i++) {
        const SymEntry *v = &s->e[i];
        uint16_t flen = (uint16_t)strlen(v->file);
        uint16_t nlen = (uint16_t)strlen(v->name);
        uint16_t slen = (uint16_t)strlen(v->sig);
        int32_t  line = (int32_t)v->line;
        int64_t  mt   = (int64_t)v->mtime;
        if (fwrite(&flen, 2, 1, f) != 1 || fwrite(v->file, 1, flen, f) != flen ||
            fwrite(&nlen, 2, 1, f) != 1 || fwrite(v->name, 1, nlen, f) != nlen ||
            fwrite(&slen, 2, 1, f) != 1 || fwrite(v->sig, 1, slen, f) != slen  ||
            fwrite(&line, 4, 1, f) != 1 || fwrite(&mt, 8, 1, f) != 1 ||
            fwrite(v->vec, sizeof(float), s->dim, f) != (size_t)s->dim) { fclose(f); unlink(tmp); return -1; }
    }
    if (fclose(f) != 0) { unlink(tmp); return -1; }
    if (rename(tmp, SYM_STORE) != 0) { unlink(tmp); return -1; }
    return 0;
}

/* ── Source-tree walk ────────────────────────────────────────────────── */

typedef struct { char *path; long mtime; } SrcFile;
typedef struct { SrcFile *v; int n, cap; } SrcList;

static bool has_src_ext(const char *name) {
    const char *dot = strrchr(name, '.');
    if (!dot) return false;
    static const char *ext[] = { ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh" };
    for (size_t i = 0; i < sizeof(ext) / sizeof(ext[0]); i++)
        if (strcmp(dot, ext[i]) == 0) return true;
    return false;
}

static bool skip_dir(const char *name) {
    if (name[0] == '.') return true;               /* .git, .basi, hidden */
    return strcmp(name, "node_modules") == 0 || strcmp(name, "build") == 0 ||
           strcmp(name, "vendor") == 0 || strcmp(name, "third_party") == 0;
}

static void srclist_walk(const char *dir, SrcList *out) {
    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0) continue;
        char path[2048];
        if ((size_t)snprintf(path, sizeof(path), "%s/%s", dir, de->d_name) >= sizeof(path)) continue;
        struct stat st;
        if (stat(path, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            if (skip_dir(de->d_name)) continue;
            if (out->n > 20000) continue;          /* runaway guard */
            srclist_walk(path, out);
        } else if (S_ISREG(st.st_mode) && has_src_ext(de->d_name) && st.st_size <= MAX_FILE_BYTES) {
            if (out->n == out->cap) { out->cap = out->cap ? out->cap * 2 : 64; out->v = realloc(out->v, out->cap * sizeof(SrcFile)); }
            /* strip a leading "./" so paths read cleanly in the warning */
            const char *p = path;
            if (p[0] == '.' && p[1] == '/') p += 2;
            out->v[out->n].path  = strdup(p);
            out->v[out->n].mtime = (long)st.st_mtime;
            out->n++;
        }
    }
    closedir(d);
}

static void srclist_free(SrcList *l) {
    for (int i = 0; i < l->n; i++) free(l->v[i].path);
    free(l->v);
    l->v = NULL; l->n = l->cap = 0;
}

/* True if the store already holds a current (matching-mtime) entry for `file`. */
static bool store_has_current(const SymStore *s, const char *file, long mtime) {
    for (size_t i = 0; i < s->n; i++)
        if (s->e[i].mtime == mtime && strcmp(s->e[i].file, file) == 0) return true;
    return false;
}

/* Reconcile the store with the current tree. Drops entries for files that are
 * gone or changed; re-embeds functions in new/changed files. Returns the count
 * of (re)embedded functions; sets *dropped. */
static int sym_sync(SymStore *s, int *dropped) {
    *dropped = 0;
    SrcList files = { NULL, 0, 0 };
    srclist_walk(".", &files);

    /* Keep pass: retain entries whose file still exists with the same mtime. */
    SymStore keep; sym_init(&keep, s->dim);
    for (size_t i = 0; i < s->n; i++) {
        SymEntry *v = &s->e[i];
        bool ok = false;
        for (int j = 0; j < files.n; j++)
            if (files.v[j].mtime == v->mtime && strcmp(files.v[j].path, v->file) == 0) { ok = true; break; }
        if (ok) {
            sym_push(&keep, v->file, v->name, v->sig, v->line, v->mtime, v->vec);
            v->file = v->name = v->sig = NULL; v->vec = NULL;   /* ownership moved */
        } else {
            (*dropped)++;
        }
    }
    for (size_t i = 0; i < s->n; i++) sym_entry_free(&s->e[i]);
    free(s->e);
    *s = keep;

    /* Add pass: (re)embed every file not already current in the store. */
    int embedded = 0;
    for (int j = 0; j < files.n; j++) {
        if (store_has_current(s, files.v[j].path, files.v[j].mtime)) continue;
        size_t flen = 0;
        char *buf = read_file_all(files.v[j].path, &flen);
        if (!buf) continue;
        int nf = 0;
        FuncDef *fs = extract_funcs(buf, flen, &nf);
        for (int k = 0; k < nf; k++) {
            if (strlen(fs[k].body) < MIN_BODY) { funcdef_free(&fs[k]); continue; }
            char *payload = build_payload(&fs[k]);
            float *vec = malloc((size_t)s->dim * sizeof(float));
            if (embed_text(payload, vec) != 0) { free(vec); free(payload); funcdef_free(&fs[k]); continue; }
            free(payload);
            sym_push(s, strdup(files.v[j].path), fs[k].name, fs[k].sig,
                     fs[k].line, files.v[j].mtime, vec);
            fs[k].name = fs[k].sig = NULL;   /* moved into the store */
            free(fs[k].body); fs[k].body = NULL;
            embedded++;
        }
        free(fs);
        free(buf);
    }
    srclist_free(&files);
    return embedded;
}

/* Best cross-file match for query vector `q` (cosine == dot on L2-normalized
 * vectors). Excludes entries in `exclude_file` (a function is never a duplicate
 * of itself). Returns store index or -1. */
static int sym_best_match(const SymStore *s, const float *q,
                          const char *exclude_file, float *out_score) {
    int best = -1; float bs = -2.0f;
    for (size_t i = 0; i < s->n; i++) {
        if (strcmp(s->e[i].file, exclude_file) == 0) continue;
        double dot = 0.0;
        const float *v = s->e[i].vec;
        for (int d = 0; d < s->dim; d++) dot += (double)v[d] * q[d];
        if ((float)dot > bs) { bs = (float)dot; best = (int)i; }
    }
    *out_score = bs;
    return best;
}

/* ── Process-lifetime singletons ─────────────────────────────────────── */

static SymStore g_store;
static bool     g_loaded  = false;
static bool     g_embed_dead = false;   /* no embedder → gate is a permanent no-op */

/* Warn-once set: a function name we've already flagged this session. Re-issuing
 * the edit then applies — the model has seen the match and made its call. */
static char **g_warned = NULL;
static int    g_warned_n = 0, g_warned_cap = 0;

static bool warned_seen(const char *name) {
    for (int i = 0; i < g_warned_n; i++) if (strcmp(g_warned[i], name) == 0) return true;
    return false;
}
static void warned_add(const char *name) {
    if (warned_seen(name)) return;
    if (g_warned_n == g_warned_cap) { g_warned_cap = g_warned_cap ? g_warned_cap * 2 : 16; g_warned = realloc(g_warned, g_warned_cap * sizeof(char *)); }
    g_warned[g_warned_n++] = strdup(name);
}

char *reuse_gate_check(const char *path, const char *replace, const char *search) {
    if (!reuse_gate_enabled()) return NULL;
    if (!replace || !*replace) return NULL;
    if (g_embed_dead) return NULL;

    if (!embed_available() || embed_init() != 0) { g_embed_dead = true; return NULL; }

    if (!g_loaded) { sym_init(&g_store, embed_dim()); sym_load(&g_store); g_loaded = true; }
    int dropped = 0;
    int embedded = sym_sync(&g_store, &dropped);
    if (embedded > 0 || dropped > 0) sym_save(&g_store);
    if (g_store.n == 0) return NULL;

    int nf = 0;
    FuncDef *fs = extract_funcs(replace, strlen(replace), &nf);
    if (nf == 0) { free(fs); return NULL; }

    float tau = env_tau();
    float *qv = malloc((size_t)g_store.dim * sizeof(float));
    StringBuf msg; sb_init(&msg);
    int hits = 0;

    for (int k = 0; k < nf; k++) {
        FuncDef *f = &fs[k];
        if (strlen(f->body) < MIN_BODY) continue;
        if (name_in_text(search, f->name)) continue;   /* editing it in place */
        if (warned_seen(f->name)) continue;             /* already flagged → let it through */

        char *payload = build_payload(f);
        int rc = embed_text(payload, qv);
        free(payload);
        if (rc != 0) continue;

        float score;
        int mi = sym_best_match(&g_store, qv, path, &score);
        if (mi < 0 || score < tau) continue;

        if (hits == 0)
            sb_append_str(&msg,
                "edit paused — reuse gate: this code may already exist in the project.\n");
        char line[1024];
        snprintf(line, sizeof(line),
            "  \xe2\x80\xa2 new function `%s` is %.2f cosine-similar to existing `%s` at %s:%d\n"
            "      %.*s\n",
            f->name, score, g_store.e[mi].name, g_store.e[mi].file, g_store.e[mi].line,
            600, g_store.e[mi].sig);
        sb_append_str(&msg, line);
        warned_add(f->name);
        hits++;
    }

    free(qv);
    for (int k = 0; k < nf; k++) funcdef_free(&fs[k]);
    free(fs);

    if (hits == 0) { sb_free(&msg); return NULL; }

    sb_append_str(&msg,
        "\nBefore writing a new function, reuse the existing one if it does the job "
        "(read it first to confirm). If you genuinely need a separate implementation, "
        "state why in one line, then re-issue this same edit — it will apply.\n");
    return sb_to_str(&msg);
}

void reuse_shutdown(void) {
    if (g_loaded) { sym_clear(&g_store); g_loaded = false; }
    for (int i = 0; i < g_warned_n; i++) free(g_warned[i]);
    free(g_warned);
    g_warned = NULL; g_warned_n = g_warned_cap = 0;
}
