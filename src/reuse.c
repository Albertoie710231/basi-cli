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
#include "embed.h"    /* only for the BASI_REUSE_SIM=embed A/B path */
#include "reuse.h"

/* ── Tunables ──────────────────────────────────────────────────────────
 * Overridable at runtime so the gate can be calibrated without a rebuild. */
#define SYM_STORE      ".basi/.symbols.bin"
#define SYM_MAGIC      0x32595342u          /* 'BSY2' little-endian */
#define SYM_VER        2u
#define MIN_BODY       60                    /* ignore trivial one-liners */
#define BODY_CAP       2000                  /* chars of a func body kept/compared */
#define MAX_FILE_BYTES (512 * 1024)          /* skip generated / vendored blobs */
#define DEFAULT_TAU_TOKEN 0.50f              /* NiCad-style token similarity */
#define DEFAULT_TAU_EMBED 0.80f              /* legacy embedding cosine */
#define TRIGRAM        3

static float env_tau(float dflt) {
    const char *e = getenv("BASI_REUSE_TAU");
    if (e && *e) { float v = (float)atof(e); if (v > 0.0f && v <= 1.0f) return v; }
    return dflt;
}

static bool env_flag(const char *name) {
    const char *e = getenv(name);
    if (!e || !*e) return false;
    char c = *e;
    return c == '1' || c == 'o' || c == 'O' || c == 't' || c == 'T' || c == 'y' || c == 'Y';
}

/* Similarity backend: default = deterministic token clone detection; embed = the
 * legacy text-embedding cosine (kept only for A/B comparison). */
static bool use_embed(void) {
    const char *m = getenv("BASI_REUSE_SIM");
    return m && strcmp(m, "embed") == 0;
}

int reuse_gate_enabled(void) { return env_flag("BASI_REUSE_GATE"); }

/* ── Heuristic C/C++ function extractor ─────────────────────────────────
 * Runs identically on files-on-disk (the index) and on an edit's REPLACE text
 * (the candidate). Not a parser: it finds top-level `name(...) { ... }`
 * definitions and skips everything else (prototypes, structs, initializers,
 * macros). See make_skeleton() for how comments/strings are neutralized. */

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

/* Same-length copy with comment and string/char-literal contents blanked to
 * spaces (newlines preserved), so brace/paren counting isn't fooled. */
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
    return len;
}

static char *collapse_range(const char *src, size_t a, size_t b) {
    StringBuf sb; sb_init(&sb);
    bool in_ws = true;
    for (size_t i = a; i < b; i++) {
        char c = src[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!in_ws) { sb_append_char(&sb, ' '); in_ws = true; }
        } else { sb_append_char(&sb, c); in_ws = false; }
    }
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

        if (sk[i] == '#') {
            while (i < len && sk[i] != '\n') {
                if (sk[i] == '\\' && i + 1 < len && sk[i + 1] == '\n') i++;
                i++;
            }
            continue;
        }
        if (sk[i] == '}' || sk[i] == ';') { i++; continue; }
        if (sk[i] == '{') { size_t e = match_brace(sk, len, i); i = (e < len) ? e + 1 : len; continue; }

        size_t start = i, j = i;
        long paren = -1;
        while (j < len && sk[j] != '{' && sk[j] != ';') {
            if (sk[j] == '(' && paren < 0) paren = (long)j;
            j++;
        }
        if (j >= len) break;
        if (sk[j] == ';') { i = j + 1; continue; }

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

/* ── NiCad-style normalization + token clone similarity ─────────────────
 * Blind-rename: every identifier that isn't a C keyword → "ID", numbers →
 * "NUM", string/char literals → "LIT"; keywords, operators and punctuation kept
 * verbatim. This defeats Type-2 (renamed-variable) duplicates that a text
 * embedding misses, while preserving the structural token sequence that keeps
 * unrelated code apart. See the arXiv notes on NiCad / SourcererCC. */

static bool is_c_keyword(const char *w, size_t n) {
    static const char *kw[] = {
        "auto","break","case","char","const","continue","default","do","double",
        "else","enum","extern","float","for","goto","if","inline","int","long",
        "register","restrict","return","short","signed","sizeof","static","struct",
        "switch","typedef","union","unsigned","void","volatile","while","bool",
        "_Bool","true","false","NULL","class","namespace","template","new","delete",
        "public","private","protected","virtual","override","nullptr","using",
    };
    for (size_t i = 0; i < sizeof(kw) / sizeof(kw[0]); i++)
        if (strlen(kw[i]) == n && memcmp(kw[i], w, n) == 0) return true;
    return false;
}

/* Space-joined normalized token stream for a function body. */
static char *ntok_string(const char *body) {
    static const char *ops[] = {
        "<<=", ">>=", "...", "->", "++", "--", "<<", ">>", "<=", ">=", "==",
        "!=", "&&", "||", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "::",
    };
    StringBuf sb; sb_init(&sb);
    const char *p = body;
    bool first = true;
    while (*p) {
        if (isspace((unsigned char)*p)) { p++; continue; }
        if (p[0] == '/' && p[1] == '/') { while (*p && *p != '\n') p++; continue; }
        if (p[0] == '/' && p[1] == '*') { p += 2; while (*p && !(p[0] == '*' && p[1] == '/')) p++; if (*p) p += 2; continue; }

        const char *emit = NULL;
        char idbuf[8];
        if (*p == '"' || *p == '\'') {
            char q = *p++;
            while (*p && *p != q) { if (*p == '\\' && p[1]) p++; p++; }
            if (*p) p++;
            emit = "LIT";
        } else if (isalpha((unsigned char)*p) || *p == '_') {
            const char *s = p;
            while (isalnum((unsigned char)*p) || *p == '_') p++;
            emit = is_c_keyword(s, (size_t)(p - s)) ? NULL : "ID";
            if (!emit) {
                size_t n = (size_t)(p - s);
                if (n >= sizeof(idbuf)) n = sizeof(idbuf) - 1;
                memcpy(idbuf, s, n); idbuf[n] = '\0'; emit = idbuf;   /* keyword verbatim */
            }
        } else if (isdigit((unsigned char)*p)) {
            while (isalnum((unsigned char)*p) || *p == '.' || *p == 'x' || *p == 'X') p++;
            emit = "NUM";
        } else {
            size_t oplen = 0;
            for (size_t k = 0; k < sizeof(ops) / sizeof(ops[0]); k++) {
                size_t l = strlen(ops[k]);
                if (strncmp(p, ops[k], l) == 0 && l > oplen) oplen = l;
            }
            if (oplen == 0) {
                char c = *p++;
                if (c == '{' || c == '}' || c == ';') continue;   /* drop formatting punctuation (brace/statement style) */
                idbuf[0] = c; idbuf[1] = '\0'; emit = idbuf;
            }
            else { size_t l = oplen < sizeof(idbuf) ? oplen : sizeof(idbuf) - 1;
                   memcpy(idbuf, p, l); idbuf[l] = '\0'; emit = idbuf; p += oplen; }
        }
        if (emit) {
            if (!first) sb_append_char(&sb, ' ');
            first = false;
            sb_append_str(&sb, emit);
        }
    }
    return sb_to_str(&sb);
}

static uint64_t fnv1a(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ULL; }
    return h;
}

static int u64cmp(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* Sorted, de-duplicated set of trigram hashes over a space-joined token
 * string. Falls back to unigram hashes when there are fewer than 3 tokens. */
static uint64_t *trigram_set(const char *ntok, size_t *out_n) {
    /* index token boundaries */
    size_t cap = 32, ntoks = 0;
    const char **tok = malloc(cap * sizeof(char *));
    size_t *len = malloc(cap * sizeof(size_t));
    const char *p = ntok;
    while (*p) {
        while (*p == ' ') p++;
        if (!*p) break;
        const char *s = p;
        while (*p && *p != ' ') p++;
        if (ntoks == cap) { cap *= 2; tok = realloc(tok, cap * sizeof(char *)); len = realloc(len, cap * sizeof(size_t)); }
        tok[ntoks] = s; len[ntoks] = (size_t)(p - s); ntoks++;
    }
    int gram = ntoks >= TRIGRAM ? TRIGRAM : 1;
    size_t cnt = ntoks >= (size_t)gram ? ntoks - gram + 1 : 0;
    uint64_t *h = malloc((cnt ? cnt : 1) * sizeof(uint64_t));
    size_t nh = 0;
    for (size_t i = 0; i + gram <= ntoks; i++) {
        char buf[256]; size_t bl = 0;
        for (int g = 0; g < gram; g++) {
            size_t l = len[i + g] < 80 ? len[i + g] : 80;
            if (bl + l + 1 < sizeof(buf)) { memcpy(buf + bl, tok[i + g], l); bl += l; buf[bl++] = 0x1f; }
        }
        h[nh++] = fnv1a(buf, bl);
    }
    free(tok); free(len);
    qsort(h, nh, sizeof(uint64_t), u64cmp);
    size_t u = 0;
    for (size_t i = 0; i < nh; i++) if (i == 0 || h[i] != h[i - 1]) h[u++] = h[i];
    *out_n = u;
    return h;
}

/* Jaccard over two sorted-unique hash sets. */
static double jaccard(const uint64_t *a, size_t na, const uint64_t *b, size_t nb) {
    if (na == 0 || nb == 0) return 0.0;
    size_t i = 0, j = 0, inter = 0;
    while (i < na && j < nb) {
        if (a[i] == b[j]) { inter++; i++; j++; }
        else if (a[i] < b[j]) i++;
        else j++;
    }
    size_t uni = na + nb - inter;
    return uni ? (double)inter / (double)uni : 0.0;
}

/* Token clone similarity between two function bodies (the primitive the gate
 * ranks on). Exposed for evaluation/reporting; deterministic, no model. */
double reuse_similarity(const char *body_a, const char *body_b) {
    if (!body_a || !body_b) return 0.0;
    char *na = ntok_string(body_a), *nb = ntok_string(body_b);
    size_t nta = 0, ntb = 0;
    uint64_t *ta = trigram_set(na, &nta), *tb = trigram_set(nb, &ntb);
    double j = jaccard(ta, nta, tb, ntb);
    free(na); free(nb); free(ta); free(tb);
    return j;
}

/* ── Symbol store (.basi/.symbols.bin) ───────────────────────────────────
 * Per-function; persisted so a fresh launch doesn't re-scan an unchanged tree.
 * Stores the (capped) body text — token similarity derives its trigram set at
 * load time, and the embed A/B path re-embeds on the fly. No GPU/model needed
 * to build or query the index in the default (token) mode. */

typedef struct {
    char     *file;
    char     *name;
    char     *sig;
    int       line;
    long      mtime;
    char     *body;      /* capped */
    uint64_t *tri;       /* derived trigram set (not persisted) */
    size_t    n_tri;
} SymEntry;

typedef struct { SymEntry *e; size_t n, cap; } SymStore;

static void sym_derive(SymEntry *v) {
    char *nt = ntok_string(v->body);
    v->tri = trigram_set(nt, &v->n_tri);
    free(nt);
}

static void sym_entry_free(SymEntry *v) {
    free(v->file); free(v->name); free(v->sig); free(v->body); free(v->tri);
    v->file = v->name = v->sig = v->body = NULL; v->tri = NULL;
}

static void sym_clear(SymStore *s) {
    for (size_t i = 0; i < s->n; i++) sym_entry_free(&s->e[i]);
    free(s->e);
    s->e = NULL; s->n = s->cap = 0;
}

static void sym_push(SymStore *s, char *file, char *name, char *sig,
                     int line, long mtime, char *body) {
    if (s->n == s->cap) { s->cap = s->cap ? s->cap * 2 : 32; s->e = realloc(s->e, s->cap * sizeof(SymEntry)); }
    SymEntry *v = &s->e[s->n++];
    v->file = file; v->name = name; v->sig = sig;
    v->line = line; v->mtime = mtime; v->body = body;
    v->tri = NULL; v->n_tri = 0;
    sym_derive(v);
}

static int sym_load(SymStore *s) {
    FILE *f = fopen(SYM_STORE, "rb");
    if (!f) return (errno == ENOENT) ? 0 : -1;
    uint32_t magic, ver, count;
    if (fread(&magic, 4, 1, f) != 1 || fread(&ver, 4, 1, f) != 1 || fread(&count, 4, 1, f) != 1) { fclose(f); return -1; }
    if (magic != SYM_MAGIC || ver != SYM_VER) { fclose(f); return -1; }
    for (uint32_t i = 0; i < count; i++) {
        uint16_t flen, nlen, slen; int32_t line; int64_t mtime; uint32_t blen;
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
        if (fread(&line, 4, 1, f) != 1 || fread(&mtime, 8, 1, f) != 1 || fread(&blen, 4, 1, f) != 1) {
            free(file); free(name); free(sig); fclose(f); return -1;
        }
        char *body = malloc(blen + 1);
        if (fread(body, 1, blen, f) != blen) { free(file); free(name); free(sig); free(body); fclose(f); return -1; }
        body[blen] = '\0';
        sym_push(s, file, name, sig, (int)line, (long)mtime, body);
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
    uint32_t magic = SYM_MAGIC, ver = SYM_VER, count = (uint32_t)s->n;
    if (fwrite(&magic, 4, 1, f) != 1 || fwrite(&ver, 4, 1, f) != 1 || fwrite(&count, 4, 1, f) != 1) { fclose(f); unlink(tmp); return -1; }
    for (size_t i = 0; i < s->n; i++) {
        const SymEntry *v = &s->e[i];
        uint16_t flen = (uint16_t)strlen(v->file), nlen = (uint16_t)strlen(v->name), slen = (uint16_t)strlen(v->sig);
        int32_t line = (int32_t)v->line; int64_t mt = (int64_t)v->mtime;
        uint32_t blen = (uint32_t)strlen(v->body);
        if (fwrite(&flen, 2, 1, f) != 1 || fwrite(v->file, 1, flen, f) != flen ||
            fwrite(&nlen, 2, 1, f) != 1 || fwrite(v->name, 1, nlen, f) != nlen ||
            fwrite(&slen, 2, 1, f) != 1 || fwrite(v->sig, 1, slen, f) != slen  ||
            fwrite(&line, 4, 1, f) != 1 || fwrite(&mt, 8, 1, f) != 1 ||
            fwrite(&blen, 4, 1, f) != 1 || fwrite(v->body, 1, blen, f) != blen) { fclose(f); unlink(tmp); return -1; }
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
    if (name[0] == '.') return true;
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
            if (out->n > 20000) continue;
            srclist_walk(path, out);
        } else if (S_ISREG(st.st_mode) && has_src_ext(de->d_name) && st.st_size <= MAX_FILE_BYTES) {
            if (out->n == out->cap) { out->cap = out->cap ? out->cap * 2 : 64; out->v = realloc(out->v, out->cap * sizeof(SrcFile)); }
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

static bool store_has_current(const SymStore *s, const char *file, long mtime) {
    for (size_t i = 0; i < s->n; i++)
        if (s->e[i].mtime == mtime && strcmp(s->e[i].file, file) == 0) return true;
    return false;
}

/* Reconcile the store with the current tree (per-file mtime). Returns the count
 * of (re)indexed functions; sets *dropped. No embedding — pure text scan. */
static int sym_sync(SymStore *s, int *dropped) {
    *dropped = 0;
    SrcList files = { NULL, 0, 0 };
    srclist_walk(".", &files);

    SymStore keep = { NULL, 0, 0 };
    for (size_t i = 0; i < s->n; i++) {
        SymEntry *v = &s->e[i];
        bool ok = false;
        for (int j = 0; j < files.n; j++)
            if (files.v[j].mtime == v->mtime && strcmp(files.v[j].path, v->file) == 0) { ok = true; break; }
        if (ok) {
            if (keep.n == keep.cap) { keep.cap = keep.cap ? keep.cap * 2 : 32; keep.e = realloc(keep.e, keep.cap * sizeof(SymEntry)); }
            keep.e[keep.n++] = *v;                 /* move (incl. derived tri) */
            v->file = v->name = v->sig = v->body = NULL; v->tri = NULL;
        } else {
            (*dropped)++;
        }
    }
    for (size_t i = 0; i < s->n; i++) sym_entry_free(&s->e[i]);
    free(s->e);
    *s = keep;

    int indexed = 0;
    for (int j = 0; j < files.n; j++) {
        if (store_has_current(s, files.v[j].path, files.v[j].mtime)) continue;
        size_t flen = 0;
        char *buf = read_file_all(files.v[j].path, &flen);
        if (!buf) continue;
        int nf = 0;
        FuncDef *fs = extract_funcs(buf, flen, &nf);
        for (int k = 0; k < nf; k++) {
            size_t blen = strlen(fs[k].body);
            if (blen < MIN_BODY) { funcdef_free(&fs[k]); continue; }
            if (blen > BODY_CAP) { fs[k].body[BODY_CAP] = '\0'; }
            sym_push(s, strdup(files.v[j].path), fs[k].name, fs[k].sig, fs[k].line, files.v[j].mtime, fs[k].body);
            fs[k].name = fs[k].sig = fs[k].body = NULL;   /* moved */
            indexed++;
        }
        free(fs);
        free(buf);
    }
    srclist_free(&files);
    return indexed;
}

/* Best cross-file match for a candidate. `exclude_file` is never matched (a
 * function is not a duplicate of itself). Token mode compares trigram sets;
 * embed mode (A/B) re-embeds bodies and takes cosine. Returns store index. */
static int sym_best_match(const SymStore *s, const FuncDef *cand,
                          const char *exclude_file, double *out_score) {
    int best = -1; double bs = -1.0;

    if (use_embed()) {
        static bool inited = false, dead = false;
        if (!dead && !inited) { if (embed_init() != 0) dead = true; inited = true; }
        if (dead) { *out_score = 0.0; return -1; }
        int dim = embed_dim();
        float *qv = malloc((size_t)dim * sizeof(float)), *ev = malloc((size_t)dim * sizeof(float));
        if (embed_text(cand->body, qv) != 0) { free(qv); free(ev); *out_score = 0.0; return -1; }
        for (size_t i = 0; i < s->n; i++) {
            if (strcmp(s->e[i].file, exclude_file) == 0) continue;
            if (embed_text(s->e[i].body, ev) != 0) continue;
            double dot = 0.0;
            for (int d = 0; d < dim; d++) dot += (double)qv[d] * ev[d];
            if (dot > bs) { bs = dot; best = (int)i; }
        }
        free(qv); free(ev);
        *out_score = bs;
        return best;
    }

    /* Token mode (default). */
    char *nt = ntok_string(cand->body);
    size_t nq = 0;
    uint64_t *q = trigram_set(nt, &nq);
    free(nt);
    for (size_t i = 0; i < s->n; i++) {
        if (strcmp(s->e[i].file, exclude_file) == 0) continue;
        double j = jaccard(q, nq, s->e[i].tri, s->e[i].n_tri);
        if (j > bs) { bs = j; best = (int)i; }
    }
    free(q);
    *out_score = bs;
    return best;
}

/* Whole-word occurrence of `name` in `hay`. */
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

/* ── Process-lifetime singletons ─────────────────────────────────────── */

static SymStore g_store;
static bool     g_loaded = false;

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

/* Lazy-load + mtime-sync the symbol index. Returns true if it holds ≥1 entry. */
static bool ensure_store(void) {
    if (!g_loaded) { g_store.e = NULL; g_store.n = g_store.cap = 0; sym_load(&g_store); g_loaded = true; }
    int dropped = 0;
    int indexed = sym_sync(&g_store, &dropped);
    if (indexed > 0 || dropped > 0) sym_save(&g_store);
    return g_store.n > 0;
}

char *reuse_gate_check(const char *path, const char *replace, const char *search) {
    if (!reuse_gate_enabled()) return NULL;
    if (!replace || !*replace) return NULL;
    if (!ensure_store()) return NULL;

    int nf = 0;
    FuncDef *fs = extract_funcs(replace, strlen(replace), &nf);
    if (nf == 0) { free(fs); return NULL; }

    float tau = env_tau(use_embed() ? DEFAULT_TAU_EMBED : DEFAULT_TAU_TOKEN);
    bool dbg = env_flag("BASI_REUSE_DEBUG");
    StringBuf msg; sb_init(&msg);
    int hits = 0;

    for (int k = 0; k < nf; k++) {
        FuncDef *f = &fs[k];
        if (strlen(f->body) < MIN_BODY) continue;
        if (name_in_text(search, f->name)) continue;
        if (warned_seen(f->name)) continue;

        double score;
        int mi = sym_best_match(&g_store, f, path, &score);
        if (dbg && mi >= 0)
            fprintf(stderr, "reuse-debug: new `%s` best=`%s` (%s:%d) score=%.3f tau=%.2f -> %s\n",
                    f->name, g_store.e[mi].name, g_store.e[mi].file, g_store.e[mi].line,
                    score, tau, score >= tau ? "PAUSE" : "allow");
        if (mi < 0 || score < tau) continue;

        if (hits == 0)
            sb_append_str(&msg, "edit paused — reuse gate: this code may already exist in the project.\n");
        char line[1024];
        snprintf(line, sizeof(line),
            "  \xe2\x80\xa2 new function `%s` closely matches existing `%s` at %s:%d (similarity %.2f)\n"
            "      %.*s\n",
            f->name, g_store.e[mi].name, g_store.e[mi].file, g_store.e[mi].line, score,
            600, g_store.e[mi].sig);
        sb_append_str(&msg, line);
        warned_add(f->name);
        hits++;
    }

    for (int k = 0; k < nf; k++) funcdef_free(&fs[k]);
    free(fs);

    if (hits == 0) { sb_free(&msg); return NULL; }

    sb_append_str(&msg,
        "\nBefore writing a new function, reuse the existing one if it does the job "
        "(read it first to confirm). If you genuinely need a separate implementation, "
        "state why in one line, then re-issue this same edit — it will apply.\n");
    return sb_to_str(&msg);
}

/* ── Deterministic reuse rewrite (output-side injection) ────────────────
 * When BASI_REUSE_AUTOFIX is on and a new function is a high-confidence,
 * exact-arity structural duplicate of an existing one, rewrite its body to a
 * single call to the existing function. This puts the reuse in the OUTPUT
 * regardless of whether the model heeds a warning — the enforcement lever a
 * decode-owning harness uniquely has. Bounded to near-identical structure +
 * matching arity so the positional-arg mapping is almost certainly correct. */

#define AUTOFIX_TAU 0.90f
#define MAX_PARAMS  12

/* Parse parameter NAMES from "ret name(p1, p2, ...)". Returns arity (0 for
 * "(void)"/"()"), or -1 on malformed. Names are malloc'd into out[]. */
static int parse_params(const char *sig, char **out, int max) {
    const char *op = strchr(sig, '(');
    const char *cp = op ? strrchr(sig, ')') : NULL;
    if (!op || !cp || cp <= op) return -1;
    int count = 0, depth = 0;
    const char *start = op + 1;
    for (const char *p = op + 1; p <= cp; p++) {
        if (p < cp && (*p == '(' || *p == '[')) depth++;
        else if (p < cp && (*p == ')' || *p == ']')) depth--;
        if (p == cp || (*p == ',' && depth == 0)) {
            const char *e = p;
            while (e > start && (isspace((unsigned char)e[-1]) || e[-1] == ']' || e[-1] == '[')) e--;
            const char *ids = e;
            while (ids > start && (isalnum((unsigned char)ids[-1]) || ids[-1] == '_')) ids--;
            size_t len = (size_t)(e - ids);
            if (len > 0) {
                char *nm = malloc(len + 1); memcpy(nm, ids, len); nm[len] = '\0';
                if (count < max) out[count++] = nm; else { free(nm); }
            }
            start = p + 1;
        }
    }
    if (count == 1 && strcmp(out[0], "void") == 0) { free(out[0]); count = 0; }
    return count;
}

/* Extract whitespace-stripped param TYPES (slice minus the trailing name and
 * any []). Returns arity (0 for void/empty); types malloc'd into out[]. */
static int parse_param_types(const char *sig, char **out, int max) {
    const char *op = strchr(sig, '(');
    const char *cp = op ? strrchr(sig, ')') : NULL;
    if (!op || !cp || cp <= op) return -1;
    int count = 0, depth = 0;
    const char *start = op + 1;
    for (const char *p = op + 1; p <= cp; p++) {
        if (p < cp && (*p == '(' || *p == '[')) depth++;
        else if (p < cp && (*p == ')' || *p == ']')) depth--;
        if (p == cp || (*p == ',' && depth == 0)) {
            const char *e = p;
            while (e > start && (isspace((unsigned char)e[-1]) || e[-1] == ']' || e[-1] == '[')) e--;
            const char *ne = e;   /* strip trailing identifier (the param name) */
            while (ne > start && (isalnum((unsigned char)ne[-1]) || ne[-1] == '_')) ne--;
            const char *tend = (ne == start) ? e : ne;   /* type-only param → whole slice */
            StringBuf b; sb_init(&b);
            for (const char *q = start; q < tend; q++)
                if (!isspace((unsigned char)*q)) sb_append_char(&b, *q);
            char *ty = sb_to_str(&b);
            if (count < max) out[count++] = ty; else free(ty);
            start = p + 1;
        }
    }
    if (count == 1 && strcmp(out[0], "void") == 0) { free(out[0]); count = 0; }
    return count;
}

/* Whitespace-stripped return type of a signature (storage-class kept; used only
 * for equality between two candidates so it cancels out). malloc'd. */
static char *return_type(const char *sig) {
    const char *op = strchr(sig, '(');
    if (!op) return strdup("");
    const char *e = op;
    while (e > sig && isspace((unsigned char)e[-1])) e--;
    while (e > sig && (isalnum((unsigned char)e[-1]) || e[-1] == '_')) e--;   /* skip fn name */
    StringBuf b; sb_init(&b);
    for (const char *q = sig; q < e; q++)
        if (!isspace((unsigned char)*q)) sb_append_char(&b, *q);
    return sb_to_str(&b);
}

/* Do candidate and helper have identical param types and return type? Same arity
 * is not enough — two enum→name mappers over different enums have matching arity
 * but incompatible types, and substituting one would not type-check. */
static bool types_compatible(const char *cand_sig, const char *helper_sig) {
    char *ct[MAX_PARAMS], *ht[MAX_PARAMS];
    int cn = parse_param_types(cand_sig, ct, MAX_PARAMS);
    int hn = parse_param_types(helper_sig, ht, MAX_PARAMS);
    bool ok = (cn >= 0 && cn == hn);
    for (int i = 0; i < cn && ok; i++)
        if (i >= hn || strcmp(ct[i], ht[i]) != 0) ok = false;
    if (ok) {
        char *cr = return_type(cand_sig), *hr = return_type(helper_sig);
        if (strcmp(cr, hr) != 0) ok = false;
        free(cr); free(hr);
    }
    for (int i = 0; i < cn; i++) free(ct[i]);
    for (int i = 0; i < hn; i++) free(ht[i]);
    return ok;
}

/* Does the signature's return type end in a bare `void` (not `void*`)? */
static bool sig_returns_void(const char *sig) {
    const char *op = strchr(sig, '(');
    if (!op) return false;
    const char *e = op;
    while (e > sig && isspace((unsigned char)e[-1])) e--;
    while (e > sig && (isalnum((unsigned char)e[-1]) || e[-1] == '_')) e--;   /* skip fn name */
    while (e > sig && isspace((unsigned char)e[-1])) e--;                     /* skip ws */
    if (e - sig >= 4 && strncmp(e - 4, "void", 4) == 0) {
        char before = (e - 4 > sig) ? e[-5] : ' ';
        return !(isalnum((unsigned char)before) || before == '_' || before == '*');
    }
    return false;
}

/* Derive "<dir>/<stem>.h" from a .c/.cpp/... helper file path. malloc'd/NULL. */
static char *header_for(const char *srcfile) {
    const char *slash = strrchr(srcfile, '/');
    const char *base = slash ? slash + 1 : srcfile;
    const char *dot = strrchr(base, '.');
    if (!dot) return NULL;
    size_t stem = (size_t)(dot - srcfile);
    char *h = malloc(stem + 3);
    memcpy(h, srcfile, stem); h[stem] = '.'; h[stem + 1] = 'h'; h[stem + 2] = '\0';
    return h;
}

/* Does header `hpath` exist and declare `name`? Cheap reachability check so we
 * never inject a call the target file can't resolve. */
static bool header_declares(const char *hpath, const char *name) {
    size_t n = 0;
    char *buf = read_file_all(hpath, &n);
    if (!buf) return false;
    bool ok = name_in_text(buf, name);
    free(buf);
    return ok;
}

/* Replace the first occurrence of `needle` in `hay` with `repl` (malloc'd). */
static char *str_replace_once(const char *hay, const char *needle, const char *repl) {
    const char *pos = strstr(hay, needle);
    if (!pos) return NULL;
    size_t pre = (size_t)(pos - hay), nlen = strlen(needle);
    StringBuf sb; sb_init(&sb);
    sb_append(&sb, hay, pre);
    sb_append_str(&sb, repl);
    sb_append_str(&sb, hay + pre + nlen);
    return sb_to_str(&sb);
}

/* ── Stage-2 semantic verification: LLVM IR equivalence ─────────────────
 * A token match says two functions LOOK alike; only the compiler can say they
 * ARE equivalent. `clang -O2` SSA form erases variable names (renamed clones →
 * identical IR) but preserves constants (SO_SNDTIMEO != SO_RCVTIMEO → different
 * IR), so an IR-equal check refuses behaviourally-distinct twins that structure
 * alone can't tell apart. Sound: IR-equal ⇒ safe to substitute; any difference
 * (or a failure to compile) ⇒ refuse. */

/* Extract the `define …@name(…){ … }` block and normalize: strip comments,
 * metadata (!…) and attribute-group refs (#N), drop linkage/param noise,
 * canonicalize the function's own name to @SELF, collapse whitespace. clang's
 * canonical SSA numbering already makes equivalent functions share register
 * numbers, so no renaming is needed. Returns malloc'd or NULL. */
static char *ir_extract_normalize(const char *ir, const char *name) {
    char nd[256];
    int ndl = snprintf(nd, sizeof(nd), "@%s(", name);   /* nd = "@name(" ; "@name" = ndl-1 chars */
    if (ndl <= 2 || ndl >= (int)sizeof(nd)) return NULL;

    const char *def = NULL, *p = ir;
    while ((p = strstr(p, "define ")) != NULL) {
        if (p == ir || p[-1] == '\n') {
            const char *brace = strchr(p, '{');
            const char *hit = strstr(p, nd);
            if (brace && hit && hit < brace) { def = p; break; }
        }
        p += 7;
    }
    if (!def) return NULL;
    const char *end = strstr(def, "\n}");
    if (!end) return NULL;
    const char *stop = end + 2;

    StringBuf out; sb_init(&out);
    const char *line = def;
    while (line < stop) {
        const char *le = memchr(line, '\n', (size_t)(stop - line));
        if (!le) le = stop;
        const char *semi = memchr(line, ';', (size_t)(le - line));
        const char *lend = semi ? semi : le;
        StringBuf lb; sb_init(&lb); bool first = true;
        const char *q = line;
        while (q < lend) {
            while (q < lend && isspace((unsigned char)*q)) q++;
            if (q >= lend) break;
            const char *ts = q;
            while (q < lend && !isspace((unsigned char)*q)) q++;
            size_t tl = (size_t)(q - ts);
            if (ts[0] == '!' || ts[0] == '#') continue;
            if ((tl == 9  && !memcmp(ts, "dso_local", 9)) ||
                (tl == 18 && !memcmp(ts, "local_unnamed_addr", 18)) ||
                (tl == 7  && !memcmp(ts, "noundef", 7)) ||
                (tl == 3  && !memcmp(ts, "nsw", 3)) ||
                (tl == 3  && !memcmp(ts, "nuw", 3))) continue;
            if (!first) sb_append_char(&lb, ' ');
            first = false;
            for (size_t i = 0; i < tl; ) {
                if (ts[i] == '@' && i + (size_t)(ndl - 1) <= tl &&
                    !memcmp(ts + i, nd, (size_t)(ndl - 1))) {
                    sb_append_str(&lb, "@SELF");
                    i += (size_t)(ndl - 1);
                } else { sb_append_char(&lb, ts[i]); i++; }
            }
        }
        char *ls = sb_to_str(&lb);
        if (*ls) { sb_append_str(&out, ls); sb_append_char(&out, '\n'); }
        free(ls);
        line = (le < stop) ? le + 1 : stop;
    }
    return sb_to_str(&out);
}

static char *compile_ir(const char *cfile, const char *iflags) {
    const char *cc = getenv("BASI_REUSE_CLANG");
    if (!cc || !*cc) cc = "clang";
    StringBuf cmd; sb_init(&cmd);
    sb_append_str(&cmd, cc);
    sb_append_str(&cmd, " -O2 -S -emit-llvm -Wno-everything ");
    sb_append_str(&cmd, iflags);
    sb_append_str(&cmd, " -o - ");
    sb_append_str(&cmd, cfile);
    sb_append_str(&cmd, " 2>/dev/null");
    char *cmdstr = sb_to_str(&cmd);   /* null-terminated; owns the buffer */
    char *ir = run_command_timeout(cmdstr, 8u * 1024 * 1024, 25, NULL);
    free(cmdstr);
    if (ir && !*ir) { free(ir); ir = NULL; }
    return ir;
}

static void dir_of(const char *path, char *buf, size_t n) {
    const char *slash = strrchr(path, '/');
    if (!slash) { snprintf(buf, n, "."); return; }
    size_t len = (size_t)(slash - path);
    if (len == 0) { snprintf(buf, n, "/"); return; }
    if (len >= n) len = n - 1;
    memcpy(buf, path, len); buf[len] = '\0';
}

/* True iff the candidate function (as it stands in `cand_src`) compiles to IR
 * byte-equal to the helper — i.e. they are genuinely equivalent, so replacing
 * the candidate's body with a call to the helper is behaviour-preserving. */
static bool ir_equivalent(const char *cand_src, const char *cand_name, const char *cand_path,
                          const char *helper_file, const char *helper_name) {
    if (kb_ensure_dirs() != 0) return false;
    const char *tmp = ".basi/.reuse_cand.c";
    FILE *f = fopen(tmp, "w");
    if (!f) return false;
    fwrite(cand_src, 1, strlen(cand_src), f);
    fclose(f);

    char hd[1024], cd[1024];
    dir_of(helper_file, hd, sizeof(hd));
    dir_of(cand_path, cd, sizeof(cd));
    StringBuf iflags; sb_init(&iflags);
    sb_append_str(&iflags, "-I. -I"); sb_append_str(&iflags, hd);
    sb_append_str(&iflags, " -I");    sb_append_str(&iflags, cd);

    char *ir_c = compile_ir(tmp, iflags.data);
    char *ir_h = compile_ir(helper_file, iflags.data);
    sb_free(&iflags);
    unlink(tmp);

    bool eq = false;
    char *nc = NULL, *nh = NULL;
    if (ir_c && ir_h) {
        nc = ir_extract_normalize(ir_c, cand_name);
        nh = ir_extract_normalize(ir_h, helper_name);
        if (nc && nh) eq = (strcmp(nc, nh) == 0);
    }
    if (env_flag("BASI_REUSE_DEBUG") && (!ir_c || !ir_h))
        fprintf(stderr, "ir-equiv: compile failed (cand=%d helper=%d) — cannot verify, refusing\n",
                ir_c != NULL, ir_h != NULL);
    free(nc); free(nh);
    free(ir_c); free(ir_h);
    return eq;
}

char *reuse_gate_autofix(const char *path, const char *replace, const char *search) {
    if (!reuse_gate_enabled() || !env_flag("BASI_REUSE_AUTOFIX")) return NULL;
    if (!replace || !*replace) return NULL;
    if (!ensure_store()) return NULL;

    int nf = 0;
    FuncDef *fs = extract_funcs(replace, strlen(replace), &nf);
    if (nf == 0) { free(fs); return NULL; }

    bool dbg = env_flag("BASI_REUSE_DEBUG");
    char *result = NULL;   /* built lazily from `replace` once a fix applies */
    int fixes = 0;

    for (int k = 0; k < nf; k++) {
        FuncDef *f = &fs[k];
        if (strlen(f->body) < MIN_BODY) continue;
        if (name_in_text(search, f->name)) continue;

        double score;
        int mi = sym_best_match(&g_store, f, path, &score);
        if (mi < 0 || score < AUTOFIX_TAU) continue;

        char *cparams[MAX_PARAMS], *hparams[MAX_PARAMS];
        int ca = parse_params(f->sig, cparams, MAX_PARAMS);
        int ha = parse_params(g_store.e[mi].sig, hparams, MAX_PARAMS);
        for (int i = 0; i < ha; i++) free(hparams[i]);
        if (ca < 0 || ca != ha) { for (int i = 0; i < ca; i++) free(cparams[i]); continue; }

        /* Type-compatibility guard: same arity is not enough (two enum→name
         * mappers over different enums match on arity but not types). */
        if (!types_compatible(f->sig, g_store.e[mi].sig)) {
            for (int i = 0; i < ca; i++) free(cparams[i]); continue;
        }

        /* Stage-2 semantic guard: structural + type match still can't tell a
         * behavioural twin (anetRecvTimeout vs anetSendTimeout — same shape,
         * one differing constant) from a true clone. Only substitute when the
         * compiler certifies the two functions are IR-equivalent. Skippable via
         * BASI_REUSE_NOVERIFY for A/B, but that reinstates the unsafe path. */
        if (!env_flag("BASI_REUSE_NOVERIFY") &&
            !ir_equivalent(replace, f->name, path, g_store.e[mi].file, g_store.e[mi].name)) {
            if (dbg) fprintf(stderr, "reuse-autofix: `%s` not IR-equivalent to `%s` — refusing rewrite\n",
                             f->name, g_store.e[mi].name);
            for (int i = 0; i < ca; i++) free(cparams[i]); continue;
        }

        /* Reachability guard: only rewrite if the helper has a header that
         * declares it — otherwise the injected call would not compile. */
        char *hdr = header_for(g_store.e[mi].file);
        if (!hdr || !header_declares(hdr, g_store.e[mi].name)) {
            free(hdr); for (int i = 0; i < ca; i++) free(cparams[i]); continue;
        }

        StringBuf b; sb_init(&b);
        sb_append_str(&b, "{\n    ");
        if (!sig_returns_void(f->sig)) sb_append_str(&b, "return ");
        sb_append_str(&b, g_store.e[mi].name);
        sb_append_char(&b, '(');
        for (int i = 0; i < ca; i++) { if (i) sb_append_str(&b, ", "); sb_append_str(&b, cparams[i]); free(cparams[i]); }
        sb_append_str(&b, ");\n}");
        char *reuse_body = sb_to_str(&b);

        const char *base = result ? result : replace;
        char *rewritten = str_replace_once(base, f->body, reuse_body);
        free(reuse_body);
        if (rewritten) {
            /* Ensure the helper's header is included so the call resolves. */
            const char *slash = strrchr(hdr, '/');
            const char *hbase = slash ? slash + 1 : hdr;
            char inc[300]; snprintf(inc, sizeof(inc), "#include \"%s\"", hbase);
            if (!strstr(rewritten, inc)) {
                StringBuf s; sb_init(&s);
                sb_append_str(&s, inc); sb_append_char(&s, '\n');
                sb_append_str(&s, rewritten);
                free(rewritten); rewritten = sb_to_str(&s);
            }
            free(result); result = rewritten; fixes++;
            if (dbg) fprintf(stderr, "reuse-autofix: rewrote `%s` -> call `%s` (%s:%d, score %.2f)\n",
                             f->name, g_store.e[mi].name, g_store.e[mi].file, g_store.e[mi].line, score);
        }
        free(hdr);
    }

    for (int k = 0; k < nf; k++) funcdef_free(&fs[k]);
    free(fs);
    return fixes ? result : (free(result), (char *)NULL);
}

void reuse_shutdown(void) {
    if (g_loaded) { sym_clear(&g_store); g_loaded = false; }
    for (int i = 0; i < g_warned_n; i++) free(g_warned[i]);
    free(g_warned);
    g_warned = NULL; g_warned_n = g_warned_cap = 0;
}
