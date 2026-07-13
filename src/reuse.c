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
#define SYM_VER        3u                    /* v3 adds a per-entry language byte */
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

/* ── Language backends ──────────────────────────────────────────────────
 * A language is pure configuration: which extensions to index, the keyword list
 * that makes the clone detector rename-insensitive, its comment/string rules,
 * and a Stage-2 verify tier (compiler IR for compiled langs, parser AST for
 * dynamic ones). Adding a language = one entry here + its tool on PATH. */

typedef enum { LANG_C = 0, LANG_CPP = 1, LANG_PY = 2, LANG_JS = 3, LANG_GO = 4,
               LANG_TS = 5, LANG_UNKNOWN = 255 } LangId;
/* Stage-2 verifier: NONE (detect-only, autofix refuses), compiler IR, or a
 * language-specific AST equivalence check (the ast_verify function pointer). */
typedef enum { VERIFY_NONE = 0, VERIFY_IR, VERIFY_AST } VerifyTier;

/* An AST-equivalence verifier: 1 iff the two whole-function sources are
 * semantically equivalent (renamed clone), 0 otherwise / on any failure. */
typedef bool (*AstVerifyFn)(const char *cand_body, const char *helper_body);
static bool py_ast_equivalent(const char *cand_body, const char *helper_body);
static bool go_ast_equivalent(const char *cand_body, const char *helper_body);
static bool ts_ast_equivalent(const char *cand_body, const char *helper_body);

typedef struct {
    LangId       id;
    const char  *name;         /* value ctags reports in the "language" field */
    const char **exts;         /* NULL-terminated extension list (with dot) */
    const char **keywords;     /* NULL-terminated; kept verbatim in normalization */
    const char  *line_comment; /* line-comment marker, or NULL */
    const char  *block_open;   /* block-comment open marker, or NULL */
    const char  *block_close;  /* block-comment close marker, or NULL */
    VerifyTier   verify;       /* how Stage-2 certifies equivalence for autofix */
    AstVerifyFn  ast_verify;   /* used iff verify == VERIFY_AST; else NULL */
    bool         brace_lang;   /* true: C-style { … } bodies; false: Python */
    const char  *import_fmt;   /* autofix reachability line; 2 %s = module,name */
} LangBackend;

/* ctags language/kind flags — one invocation handles every backend, ctags
 * auto-detects each file's language by extension and tags it accordingly. */
#define CTAGS_ARGS "--output-format=json --fields=+neSl " \
    "--languages=C,C++,Python,JavaScript,Go,TypeScript " \
    "--kinds-C=f --kinds-C++=f --kinds-Python=fm --kinds-JavaScript=fm " \
    "--kinds-Go=f --kinds-TypeScript=fm"

static const char *C_EXTS[]   = { ".c", ".h", NULL };
static const char *CPP_EXTS[] = { ".cpp", ".cc", ".cxx", ".hpp", ".hh", NULL };
static const char *PY_EXTS[]  = { ".py", NULL };
static const char *JS_EXTS[]  = { ".js", ".mjs", ".cjs", NULL };
static const char *GO_EXTS[]  = { ".go", NULL };
static const char *TS_EXTS[]  = { ".ts", ".tsx", ".mts", ".cts", NULL };

static const char *C_KW[] = {
    "auto","break","case","char","const","continue","default","do","double",
    "else","enum","extern","float","for","goto","if","inline","int","long",
    "register","restrict","return","short","signed","sizeof","static","struct",
    "switch","typedef","union","unsigned","void","volatile","while","bool",
    "_Bool","true","false","NULL","class","namespace","template","new","delete",
    "public","private","protected","virtual","override","nullptr","using", NULL,
};
static const char *PY_KW[] = {
    "False","None","True","and","as","assert","async","await","break","class",
    "continue","def","del","elif","else","except","finally","for","from","global",
    "if","import","in","is","lambda","nonlocal","not","or","pass","raise","return",
    "try","while","with","yield","match","case", NULL,
};
static const char *JS_KW[] = {
    "break","case","catch","class","const","continue","debugger","default",
    "delete","do","else","export","extends","finally","for","function","if",
    "import","in","instanceof","new","return","super","switch","this","throw",
    "try","typeof","var","void","while","with","yield","let","static","async",
    "await","of","true","false","null","undefined", NULL,
};
static const char *GO_KW[] = {
    "break","case","chan","const","continue","default","defer","else",
    "fallthrough","for","func","go","goto","if","import","interface","map",
    "package","range","return","select","struct","switch","type","var",
    "true","false","nil","iota", NULL,
};
static const char *TS_KW[] = {   /* JS keywords + the common TypeScript ones */
    "break","case","catch","class","const","continue","debugger","default",
    "delete","do","else","export","extends","finally","for","function","if",
    "import","in","instanceof","new","return","super","switch","this","throw",
    "try","typeof","var","void","while","with","yield","let","static","async",
    "await","of","true","false","null","undefined",
    "abstract","any","as","asserts","boolean","declare","enum","implements",
    "interface","is","keyof","namespace","never","number","readonly","string",
    "type","unknown","public","private","protected","override","satisfies","infer", NULL,
};

static const LangBackend BACKENDS[] = {
    { LANG_C,   "C",          C_EXTS,   C_KW,  "//", "/*", "*/", VERIFY_IR,   NULL,               true,  NULL },
    { LANG_CPP, "C++",        CPP_EXTS, C_KW,  "//", "/*", "*/", VERIFY_IR,   NULL,               true,  NULL },
    { LANG_PY,  "Python",     PY_EXTS,  PY_KW, "#",  NULL, NULL, VERIFY_AST,  py_ast_equivalent,  false, "from %s import %s" },
    { LANG_JS,  "JavaScript", JS_EXTS,  JS_KW, "//", "/*", "*/", VERIFY_NONE, NULL,               true,  NULL },
    { LANG_GO,  "Go",         GO_EXTS,  GO_KW, "//", "/*", "*/", VERIFY_AST,  go_ast_equivalent,  true,  NULL },
    { LANG_TS,  "TypeScript", TS_EXTS,  TS_KW, "//", "/*", "*/", VERIFY_AST,  ts_ast_equivalent,  true,  NULL },
};
#define N_BACKENDS (sizeof(BACKENDS) / sizeof(BACKENDS[0]))

static const LangBackend *backend_by_id(LangId id) {
    for (size_t i = 0; i < N_BACKENDS; i++) if (BACKENDS[i].id == id) return &BACKENDS[i];
    return NULL;
}
static const LangBackend *backend_by_name(const char *ctags_lang) {
    if (!ctags_lang) return NULL;
    for (size_t i = 0; i < N_BACKENDS; i++)
        if (strcmp(BACKENDS[i].name, ctags_lang) == 0) return &BACKENDS[i];
    return NULL;
}
/* LangId for a path (by extension); LANG_UNKNOWN if unsupported. Used to filter
 * the tree walk and choose the candidate temp file's extension. The definitive
 * per-function language always comes from ctags' own detection. */
static LangId lang_of_path(const char *path) {
    const char *dot = strrchr(path, '.');
    if (!dot) return LANG_UNKNOWN;
    for (size_t i = 0; i < N_BACKENDS; i++)
        for (const char **e = BACKENDS[i].exts; *e; e++)
            if (strcmp(dot, *e) == 0) return BACKENDS[i].id;
    return LANG_UNKNOWN;
}

/* ── ctags-based function extractor ─────────────────────────────────────
 * Language-agnostic: shells out to Universal Ctags, which reports each function
 * with a start line, an end line and a parameter signature. The body is the
 * whole-function text (lines [start..end], signature included) — uniform across
 * braced and indentation-based languages. Runs identically on files-on-disk
 * (the index) and on an edit's REPLACE text written to a temp file. */

typedef struct {
    char  *name;
    char  *sig;    /* reconstructed "<ret> name(params)" for the autofix parsers */
    char  *body;   /* whole-function text, lines [start..end] */
    int    line;   /* 1-based start line */
    LangId lang;
} FuncDef;

static void funcdef_free(FuncDef *f) {
    free(f->name); free(f->sig); free(f->body);
    f->name = f->sig = f->body = NULL;
}

static char *slice_dup(const char *src, size_t a, size_t b) {
    size_t n = b - a;
    char *o = malloc(n + 1);
    memcpy(o, src + a, n);
    o[n] = '\0';
    return o;
}

/* Byte offset of the start of 1-based line `ln`; `n` (EOF) if past the end. */
static size_t line_offset(const char *src, size_t n, int ln) {
    if (ln <= 1) return 0;
    int cur = 1;
    for (size_t i = 0; i < n; i++)
        if (src[i] == '\n') { cur++; if (cur == ln) return i + 1; }
    return n;
}

/* Tiny scalar readers over one JSON tag line (must be NUL-terminated at its
 * newline first, so strstr stays within the line). */
static char *j_str(const char *o, const char *k) {
    char pat[48]; int pl = snprintf(pat, sizeof(pat), "\"%s\":", k);
    if (pl <= 0 || pl >= (int)sizeof(pat)) return NULL;
    const char *x = strstr(o, pat); if (!x) return NULL;
    const char *q = x + pl; while (*q == ' ') q++;
    if (*q != '"') return NULL;
    q++;
    StringBuf s; sb_init(&s);
    while (*q && *q != '"') {
        if (*q == '\\' && q[1]) { q++; sb_append_char(&s, *q == 'n' ? '\n' : (*q == 't' ? '\t' : *q)); }
        else sb_append_char(&s, *q);
        q++;
    }
    return sb_to_str(&s);
}
static int j_int(const char *o, const char *k) {
    char pat[48]; int pl = snprintf(pat, sizeof(pat), "\"%s\":", k);
    if (pl <= 0 || pl >= (int)sizeof(pat)) return -1;
    const char *x = strstr(o, pat); if (!x) return -1;
    const char *q = x + pl; while (*q == ' ') q++;
    return atoi(q);
}

/* Offset just past the '}' that closes the first '{' at or after `from`,
 * skipping braces inside strings, char literals and line/block comments. Used
 * as a fallback body-extent for brace languages when ctags omits the `end`
 * field (e.g. its JavaScript parser). Returns `flen` if unbalanced. */
static size_t brace_extent(const char *src, size_t flen, size_t from) {
    size_t i = from;
    while (i < flen && src[i] != '{') i++;
    if (i >= flen) return flen;
    int depth = 0;
    enum { CODE, STR, CH, LC, BC } st = CODE;
    char q = 0;
    for (; i < flen; i++) {
        char c = src[i], d = (i + 1 < flen) ? src[i + 1] : '\0';
        switch (st) {
        case CODE:
            if (c == '/' && d == '/') { st = LC; i++; }
            else if (c == '/' && d == '*') { st = BC; i++; }
            else if (c == '"' || c == '\'') { st = (c == '"') ? STR : CH; q = c; }
            else if (c == '{') depth++;
            else if (c == '}') { if (--depth == 0) return i + 1; }
            break;
        case STR: case CH:
            if (c == '\\') i++;
            else if (c == q) st = CODE;
            break;
        case LC: if (c == '\n') st = CODE; break;
        case BC: if (c == '*' && d == '/') { st = CODE; i++; } break;
        }
    }
    return flen;
}

static FuncDef *ctags_extract_file(const char *path, int *out_n) {
    *out_n = 0;
    size_t flen = 0;
    char *src = read_file_all(path, &flen);
    if (!src) return NULL;

    const char *ct = getenv("BASI_REUSE_CTAGS");
    StringBuf cmd; sb_init(&cmd);
    sb_append_str(&cmd, (ct && *ct) ? ct : "ctags");
    sb_append_str(&cmd, " " CTAGS_ARGS " -o - ");
    sb_append_str(&cmd, path);
    sb_append_str(&cmd, " 2>/dev/null");
    char *cmdstr = sb_to_str(&cmd);
    char *out = run_command(cmdstr, 8u * 1024 * 1024);
    free(cmdstr);
    if (!out) { free(src); return NULL; }

    FuncDef *arr = NULL; int n = 0, cap = 0;
    char *line = out;
    while (*line) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';
        if (strstr(line, "\"_type\"")) {
            char *name = j_str(line, "name");
            int   ls   = j_int(line, "line");
            int   le   = j_int(line, "end");
            char *sig  = j_str(line, "signature");
            char *lang = j_str(line, "language");
            char *tref = j_str(line, "typeref");
            if (name && ls > 0) {
                const LangBackend *be = backend_by_name(lang);
                size_t a = line_offset(src, flen, ls);
                size_t b;
                if (le >= ls) {
                    b = line_offset(src, flen, le + 1);        /* ctags gave end */
                } else if (be && be->brace_lang) {
                    b = brace_extent(src, flen, a);            /* fallback (e.g. JS) */
                } else {
                    b = line_offset(src, flen, ls + 1);        /* single line */
                }
                /* Rebuild a C-style "<ret> name(params)" signature so the
                 * autofix param/type/void parsers keep working unchanged. For
                 * dynamic langs (no typeref) the return type is simply absent. */
                StringBuf sb; sb_init(&sb);
                if (tref) {
                    const char *colon = strchr(tref, ':');
                    sb_append_str(&sb, colon ? colon + 1 : tref);
                    sb_append_char(&sb, ' ');
                }
                sb_append_str(&sb, name);
                sb_append_str(&sb, sig ? sig : "()");
                if (n >= cap) { cap = cap ? cap * 2 : 8; arr = realloc(arr, cap * sizeof(FuncDef)); }
                arr[n].name = name; name = NULL;
                arr[n].sig  = sb_to_str(&sb);
                arr[n].body = slice_dup(src, a, b);
                arr[n].line = ls;
                arr[n].lang = be ? be->id : LANG_UNKNOWN;
                n++;
            }
            free(name); free(sig); free(lang); free(tref);
        }
        if (!nl) break;
        line = nl + 1;
    }
    free(out); free(src);
    *out_n = n;
    return arr;
}

/* Extract the functions an edit ADDS by writing its REPLACE text to a temp file
 * (extension chosen from the target path's language) and ctags-ing it. */
static FuncDef *extract_candidate(const char *path, const char *replace, int *out_n) {
    *out_n = 0;
    const LangBackend *be = backend_by_id(lang_of_path(path));
    const char *ext = be ? be->exts[0] : ".c";
    if (kb_ensure_dirs() != 0) return NULL;
    char tmp[256];
    snprintf(tmp, sizeof(tmp), ".basi/.reuse_ext%s", ext);
    FILE *f = fopen(tmp, "w");
    if (!f) return NULL;
    fwrite(replace, 1, strlen(replace), f);
    fclose(f);
    FuncDef *fs = ctags_extract_file(tmp, out_n);
    unlink(tmp);
    return fs;
}

/* ── NiCad-style normalization + token clone similarity ─────────────────
 * Blind-rename: every identifier that isn't one of the language's keywords →
 * "ID", numbers → "NUM", string/char literals → "LIT"; keywords, operators and
 * punctuation kept verbatim. This defeats Type-2 (renamed-variable) duplicates
 * that a text embedding misses, while preserving the structural token sequence
 * that keeps unrelated code apart. See the arXiv notes on NiCad / SourcererCC.
 * The keyword list and comment/string rules come from the LangBackend, so the
 * same code normalizes C, C++, Python, … . */

static bool kw_is(const LangBackend *be, const char *w, size_t n) {
    for (const char **k = be->keywords; *k; k++)
        if (strlen(*k) == n && memcmp(*k, w, n) == 0) return true;
    return false;
}

/* Space-joined normalized token stream for a function body, per language. */
static char *ntok_string(const LangBackend *be, const char *body) {
    static const char *ops[] = {
        "<<=", ">>=", "...", "//", "**", "->", "++", "--", "<<", ">>", "<=",
        ">=", "==", "!=", "&&", "||", "+=", "-=", "*=", "/=", "%=", "&=", "|=",
        "^=", "::", ":=",
    };
    size_t lc = be->line_comment ? strlen(be->line_comment) : 0;
    size_t bo = be->block_open   ? strlen(be->block_open)   : 0;
    size_t bc = be->block_close  ? strlen(be->block_close)  : 0;
    StringBuf sb; sb_init(&sb);
    const char *p = body;
    bool first = true;
    while (*p) {
        if (isspace((unsigned char)*p)) { p++; continue; }
        if (lc && strncmp(p, be->line_comment, lc) == 0) { while (*p && *p != '\n') p++; continue; }
        if (bo && strncmp(p, be->block_open, bo) == 0) {
            p += bo; while (*p && !(bc && strncmp(p, be->block_close, bc) == 0)) p++; if (*p) p += bc; continue;
        }

        const char *emit = NULL;
        char idbuf[8];
        if (*p == '"' || *p == '\'') {
            char q = *p;
            if (!be->brace_lang && p[1] == q && p[2] == q) {           /* python triple-quote */
                p += 3; while (*p && !(p[0] == q && p[1] == q && p[2] == q)) p++; if (*p) p += 3;
            } else {
                p++; while (*p && *p != q) { if (*p == '\\' && p[1]) p++; p++; } if (*p) p++;
            }
            emit = "LIT";
        } else if (isalpha((unsigned char)*p) || *p == '_') {
            const char *s = p;
            while (isalnum((unsigned char)*p) || *p == '_') p++;
            size_t n = (size_t)(p - s);
            if (kw_is(be, s, n)) {
                if (n >= sizeof(idbuf)) n = sizeof(idbuf) - 1;
                memcpy(idbuf, s, n); idbuf[n] = '\0'; emit = idbuf;   /* keyword verbatim */
            } else emit = "ID";
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
                if (be->brace_lang && (c == '{' || c == '}' || c == ';')) continue;   /* drop brace/statement formatting */
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
    const LangBackend *be = backend_by_id(LANG_C);   /* eval harness is C */
    char *na = ntok_string(be, body_a), *nb = ntok_string(be, body_b);
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
    LangId    lang;      /* so matches never cross languages */
    char     *body;      /* capped */
    uint64_t *tri;       /* derived trigram set (not persisted) */
    size_t    n_tri;
} SymEntry;

typedef struct { SymEntry *e; size_t n, cap; } SymStore;

static void sym_derive(SymEntry *v) {
    const LangBackend *be = backend_by_id(v->lang);
    if (!be) be = backend_by_id(LANG_C);   /* unknown → C rules (safe default) */
    char *nt = ntok_string(be, v->body);
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
                     int line, long mtime, LangId lang, char *body) {
    if (s->n == s->cap) { s->cap = s->cap ? s->cap * 2 : 32; s->e = realloc(s->e, s->cap * sizeof(SymEntry)); }
    SymEntry *v = &s->e[s->n++];
    v->file = file; v->name = name; v->sig = sig;
    v->line = line; v->mtime = mtime; v->lang = lang; v->body = body;
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
        uint8_t lang;
        if (fread(&line, 4, 1, f) != 1 || fread(&mtime, 8, 1, f) != 1 ||
            fread(&lang, 1, 1, f) != 1 || fread(&blen, 4, 1, f) != 1) {
            free(file); free(name); free(sig); fclose(f); return -1;
        }
        char *body = malloc(blen + 1);
        if (fread(body, 1, blen, f) != blen) { free(file); free(name); free(sig); free(body); fclose(f); return -1; }
        body[blen] = '\0';
        sym_push(s, file, name, sig, (int)line, (long)mtime, (LangId)lang, body);
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
        uint8_t lang = (uint8_t)v->lang;
        uint32_t blen = (uint32_t)strlen(v->body);
        if (fwrite(&flen, 2, 1, f) != 1 || fwrite(v->file, 1, flen, f) != flen ||
            fwrite(&nlen, 2, 1, f) != 1 || fwrite(v->name, 1, nlen, f) != nlen ||
            fwrite(&slen, 2, 1, f) != 1 || fwrite(v->sig, 1, slen, f) != slen  ||
            fwrite(&line, 4, 1, f) != 1 || fwrite(&mt, 8, 1, f) != 1 ||
            fwrite(&lang, 1, 1, f) != 1 ||
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
    return lang_of_path(name) != LANG_UNKNOWN;
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
        int nf = 0;
        FuncDef *fs = ctags_extract_file(files.v[j].path, &nf);
        for (int k = 0; k < nf; k++) {
            size_t blen = strlen(fs[k].body);
            if (blen < MIN_BODY) { funcdef_free(&fs[k]); continue; }
            if (blen > BODY_CAP) { fs[k].body[BODY_CAP] = '\0'; }
            sym_push(s, strdup(files.v[j].path), fs[k].name, fs[k].sig, fs[k].line,
                     files.v[j].mtime, fs[k].lang, fs[k].body);
            fs[k].name = fs[k].sig = fs[k].body = NULL;   /* moved */
            indexed++;
        }
        free(fs);
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
            if (s->e[i].lang != cand->lang) continue;   /* never match across languages */
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
    const LangBackend *be = backend_by_id(cand->lang);
    if (!be) be = backend_by_id(LANG_C);
    char *nt = ntok_string(be, cand->body);
    size_t nq = 0;
    uint64_t *q = trigram_set(nt, &nq);
    free(nt);
    for (size_t i = 0; i < s->n; i++) {
        if (s->e[i].lang != cand->lang) continue;       /* never match across languages */
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
    FuncDef *fs = extract_candidate(path, replace, &nf);
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

/* Parse parameter NAMES from a Go signature "ret name(a, b int, c string)":
 * the name is the LEADING identifier of each comma group (Go puts the type
 * after the name, and shares a type across a group). Returns arity or -1. */
static int parse_params_go(const char *sig, char **out, int max) {
    const char *op = strchr(sig, '(');
    const char *cp = op ? strrchr(sig, ')') : NULL;
    if (!op || !cp || cp <= op) return -1;
    int count = 0, depth = 0;
    const char *start = op + 1;
    for (const char *p = op + 1; p <= cp; p++) {
        if (p < cp && (*p == '(' || *p == '[' || *p == '{')) depth++;
        else if (p < cp && (*p == ')' || *p == ']' || *p == '}')) depth--;
        if (p == cp || (*p == ',' && depth == 0)) {
            const char *s = start;
            while (s < p && isspace((unsigned char)*s)) s++;
            const char *e = s;
            while (e < p && (isalnum((unsigned char)*e) || *e == '_')) e++;
            if (e > s) {
                size_t len = (size_t)(e - s);
                char *nm = malloc(len + 1); memcpy(nm, s, len); nm[len] = '\0';
                if (count < max) out[count++] = nm; else free(nm);
            }
            start = p + 1;
        }
    }
    return count;
}

/* Parameter names for the candidate/helper, dispatched by language. */
static int params_for(const LangBackend *be, const char *sig, char **out, int max) {
    return (be && be->id == LANG_GO) ? parse_params_go(sig, out, max)
                                     : parse_params(sig, out, max);
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

/* Append the body of module-level "attributes #n = { … }" to `sb`, so two
 * functions are compared by their actual attribute SET (memory(none),
 * nounwind, …) rather than a module-relative index. Falls back to "#n". */
static void resolve_attr(StringBuf *sb, const char *ir, int n) {
    char pat[40]; snprintf(pat, sizeof(pat), "\nattributes #%d = ", n);
    const char *a = strstr(ir, pat);
    if (a) a = strchr(a, '{');
    const char *e = a ? strchr(a, '}') : NULL;
    if (!a || !e) { char t[16]; snprintf(t, sizeof(t), "#%d", n); sb_append_str(sb, t); return; }
    sb_append(sb, a, (size_t)(e - a + 1));            /* the "{ … }" verbatim */
}

/* Append the recursively-resolved definition of metadata "!n" to `sb`,
 * canonicalizing module-relative numbering and guarding the self-referential
 * cycles LLVM uses for `distinct` loop metadata. This compares !tbaa / !range /
 * !llvm.loop by CONTENT, so a real aliasing/range difference is caught while a
 * genuine clone (identical metadata content) still matches. */
static void resolve_md(StringBuf *sb, const char *ir, int n, int *seen, int *nseen, int depth) {
    if (depth > 24) { sb_append_str(sb, "!.."); return; }
    for (int i = 0; i < *nseen; i++) if (seen[i] == n) { sb_append_str(sb, "!cyc"); return; }
    char pat[24]; snprintf(pat, sizeof(pat), "\n!%d = ", n);
    const char *d = strstr(ir, pat);
    if (!d) { char t[16]; snprintf(t, sizeof(t), "!%d", n); sb_append_str(sb, t); return; }
    d += strlen(pat);
    const char *e = strchr(d, '\n'); if (!e) e = d + strlen(d);
    if (*nseen < 64) seen[(*nseen)++] = n;
    for (const char *p = d; p < e; ) {
        if (*p == '!' && p + 1 < e && isdigit((unsigned char)p[1])) {
            resolve_md(sb, ir, atoi(p + 1), seen, nseen, depth + 1);
            p++; while (p < e && isdigit((unsigned char)*p)) p++;
        } else { sb_append_char(sb, *p); p++; }
    }
    if (*nseen > 0) (*nseen)--;
}

/* Extract the `define …@sym(…){ … }` block and normalize: strip comments and
 * linkage noise, canonicalize the function's own name to @SELF, and RESOLVE
 * attribute-group (#N) and metadata (!N) references to their content so a
 * module-relative index difference doesn't hide a real semantic one. clang's
 * canonical SSA numbering already makes equivalent functions share register
 * numbers, so no register renaming is needed. Returns malloc'd or NULL.
 *
 * The self-symbol is `@name` for C (verbatim) or the Itanium-mangled
 * `@_Z…<len><name>…` for C++; both are canonicalized to @SELF so renamed
 * clones compare equal regardless of mangling. */
static char *ir_extract_normalize(const char *ir, const char *name) {
    char nd[256];
    int ndl = snprintf(nd, sizeof(nd), "@%s(", name);   /* "@name(" */
    if (ndl <= 2 || ndl >= (int)sizeof(nd)) return NULL;

    char self_tok[256]; size_t self_len = 0;   /* the exact "@symbol" to blank */
    const char *def = NULL, *p = ir;

    /* Pass 1: exact @name( — C, or C++ with extern "C". */
    while ((p = strstr(p, "define ")) != NULL) {
        if (p == ir || p[-1] == '\n') {
            const char *brace = strchr(p, '{');
            const char *hit = strstr(p, nd);
            if (brace && hit && hit < brace) {
                def = p;
                self_len = (size_t)snprintf(self_tok, sizeof(self_tok), "@%s", name);
                break;
            }
        }
        p += 7;
    }
    /* Pass 2: mangled symbol containing "<len>name" — C++ (name mangling). */
    if (!def) {
        char lp[128];
        int lpl = snprintf(lp, sizeof(lp), "%zu%s", strlen(name), name);   /* "5clamp" */
        if (lpl > 0 && lpl < (int)sizeof(lp)) {
            p = ir;
            while ((p = strstr(p, "define ")) != NULL) {
                if (p == ir || p[-1] == '\n') {
                    const char *brace = strchr(p, '{');
                    const char *q = p;
                    while (brace && q < brace &&
                           (q = memchr(q, '@', (size_t)(brace - q))) != NULL) {
                        const char *paren = strchr(q, '(');
                        if (paren && paren < brace && (size_t)(paren - q) < sizeof(self_tok) &&
                            memmem(q, (size_t)(paren - q), lp, (size_t)lpl)) {
                            def = p; self_len = (size_t)(paren - q);
                            memcpy(self_tok, q, self_len); self_tok[self_len] = '\0';
                            break;
                        }
                        q++;
                    }
                    if (def) break;
                }
                p += 7;
            }
        }
    }
    if (!def || self_len == 0) return NULL;
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
            /* Attribute-group / metadata references: resolve to their content
             * (their numbers are module-relative, so a raw #N/!N comparison is
             * meaningless — but the SET/tree they name is what matters). A
             * named kind like `!tbaa` is kept verbatim; `!N`/`#N` are resolved. */
            if (ts[0] == '#' && tl > 1 && isdigit((unsigned char)ts[1])) {
                if (!first) sb_append_char(&lb, ' ');
                first = false;
                resolve_attr(&lb, ir, atoi(ts + 1));
                continue;
            }
            if (ts[0] == '!' && tl > 1 && isdigit((unsigned char)ts[1])) {
                if (!first) sb_append_char(&lb, ' ');
                first = false;
                int seen[64], nseen = 0;
                resolve_md(&lb, ir, atoi(ts + 1), seen, &nseen, 0);
                continue;
            }
            /* Strip only non-semantic linkage/visibility tokens. `nsw`/`nuw`
             * (no-signed/unsigned-wrap) and `noundef` are NOT stripped: they
             * carry poison/UB semantics, so two functions differing only in
             * them are not refinement-equivalent (a signed `mul nsw` may be
             * poison on overflow where an unsigned `mul` is defined). Dropping
             * them made the verifier certify such twins as equal — an unsound
             * autofix. Keeping them refuses the substitution (sound: a genuine
             * renamed clone still emits identical flags and still matches). */
            if ((tl == 9  && !memcmp(ts, "dso_local", 9)) ||
                (tl == 18 && !memcmp(ts, "local_unnamed_addr", 18))) continue;
            if (!first) sb_append_char(&lb, ' ');
            first = false;
            for (size_t i = 0; i < tl; ) {
                if (ts[i] == '@' && i + self_len <= tl &&
                    !memcmp(ts + i, self_tok, self_len)) {
                    sb_append_str(&lb, "@SELF");
                    i += self_len;
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

/* Value of a "key":"…" string pair within a bounded JSON object slice. */
static char *cc_json_str(const char *obj, size_t len, const char *key) {
    char pat[64];
    int pl = snprintf(pat, sizeof(pat), "\"%s\"", key);
    if (pl <= 0 || pl >= (int)sizeof(pat)) return NULL;
    const char *k = memmem(obj, len, pat, (size_t)pl);
    if (!k) return NULL;
    const char *p = k + pl, *end = obj + len;
    while (p < end && (*p == ' ' || *p == '\t' || *p == ':' || *p == '\n' || *p == '\r')) p++;
    if (p >= end || *p != '"') return NULL;
    p++;
    StringBuf sb; sb_init(&sb);
    while (p < end && *p != '"') {
        if (*p == '\\' && p + 1 < end) {
            p++;
            char c = *p;
            sb_append_char(&sb, c == 'n' ? '\n' : (c == 't' ? '\t' : c));
        } else sb_append_char(&sb, *p);
        p++;
    }
    return sb_to_str(&sb);
}

/* Keep only the preprocessing flags (-I/-D/-U/-std/-isystem/-include) from a
 * full compile command — the parts that give a file its include/define context. */
static char *cc_extract_flags(const char *cmd) {
    StringBuf out; sb_init(&out);
    const char *p = cmd;
    while (*p) {
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;
        const char *ts = p;
        while (*p && *p != ' ' && *p != '\t') p++;
        size_t tl = (size_t)(p - ts);
        bool keep = false, takes_arg = false;
        if (tl >= 2 && ts[0] == '-') {
            char c = ts[1];
            if (c == 'I' || c == 'D' || c == 'U') { keep = true; takes_arg = (tl == 2); }
            else if (tl >= 4 && !strncmp(ts, "-std", 4)) keep = true;
            else if (tl == 8 && !strncmp(ts, "-isystem", 8)) { keep = true; takes_arg = true; }
            else if (tl == 8 && !strncmp(ts, "-include", 8)) { keep = true; takes_arg = true; }
        }
        if (keep) {
            if (out.len) sb_append_char(&out, ' ');
            sb_append(&out, ts, tl);
            if (takes_arg) {
                while (*p == ' ' || *p == '\t') p++;
                const char *as = p;
                while (*p && *p != ' ' && *p != '\t') p++;
                if (p > as) { sb_append_char(&out, ' '); sb_append(&out, as, (size_t)(p - as)); }
            }
        }
    }
    return sb_to_str(&out);
}

/* Compile flags for `target` from ./compile_commands.json (matched by basename),
 * or NULL if there is no database or no matching entry. */
static char *cc_lookup_flags(const char *target) {
    size_t n = 0;
    char *json = read_file_all("compile_commands.json", &n);
    if (!json) return NULL;
    const char *tb = strrchr(target, '/'); tb = tb ? tb + 1 : target;
    char *result = NULL;
    const char *p = json, *end = json + n;
    while (p < end && !result) {
        const char *ob = memchr(p, '{', (size_t)(end - p));
        if (!ob) break;
        const char *oe = memchr(ob, '}', (size_t)(end - ob));
        if (!oe) break;
        size_t olen = (size_t)(oe - ob + 1);
        char *file = cc_json_str(ob, olen, "file");
        if (file) {
            const char *fb = strrchr(file, '/'); fb = fb ? fb + 1 : file;
            if (strcmp(fb, tb) == 0) {
                char *cmd = cc_json_str(ob, olen, "command");
                if (cmd) { result = cc_extract_flags(cmd); free(cmd); }
            }
            free(file);
        }
        p = oe + 1;
    }
    free(json);
    return result;
}

/* True iff the candidate function (as it stands in `cand_src`) compiles to IR
 * byte-equal to the helper — i.e. they are genuinely equivalent, so replacing
 * the candidate's body with a call to the helper is behaviour-preserving. */
static bool ir_equivalent(const char *cand_src, const char *cand_name, const char *cand_path,
                          const char *helper_file, const char *helper_name) {
    if (kb_ensure_dirs() != 0) return false;
    /* Candidate temp file must carry the language's extension so clang selects
     * the matching frontend (C vs C++) — a C++ candidate written to a .c file
     * would be miscompiled and its mangled name would never match the helper. */
    const LangBackend *cbe = backend_by_id(lang_of_path(cand_path));
    char tmp[64];
    snprintf(tmp, sizeof(tmp), ".basi/.reuse_cand%s", cbe ? cbe->exts[0] : ".c");
    FILE *f = fopen(tmp, "w");
    if (!f) return false;
    fwrite(cand_src, 1, strlen(cand_src), f);
    fclose(f);

    /* Prefer the project's real per-file flags (compile_commands.json); fall
     * back to a best-effort guess for self-contained files / no database. */
    char *ccflags = cc_lookup_flags(helper_file);
    char *guessed = NULL;
    const char *iflags;
    if (ccflags) {
        iflags = ccflags;
    } else {
        char hd[1024], cd[1024];
        dir_of(helper_file, hd, sizeof(hd));
        dir_of(cand_path, cd, sizeof(cd));
        StringBuf g; sb_init(&g);
        sb_append_str(&g, "-I. -I"); sb_append_str(&g, hd);
        sb_append_str(&g, " -I");    sb_append_str(&g, cd);
        guessed = sb_to_str(&g);
        iflags = guessed;
    }
    if (env_flag("BASI_REUSE_DEBUG"))
        fprintf(stderr, "ir-equiv flags [%s]: %s\n", ccflags ? "compile_commands.json" : "guessed", iflags);

    char *ir_c = compile_ir(tmp, iflags);
    char *ir_h = compile_ir(helper_file, iflags);
    free(ccflags); free(guessed);
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

/* ── Stage-2 semantic verification: Python AST equivalence ──────────────
 * The dynamic-language analog of IR equivalence. Renaming only the function's
 * own parameters (its locals) — not called-function names or constants — then
 * comparing the normalized AST dumps: a renamed clone matches, a constant twin
 * (21 vs 20) or a differing call does not. Uses only the builtin `ast` module,
 * so the sole dependency is python3 itself. */
#define PY_AST_SCRIPT \
    "import ast,sys,textwrap\n" \
    "def norm(t):\n" \
    " fn=ast.parse(textwrap.dedent(t)).body[0]\n" \
    " params={a.arg for a in fn.args.args}\n" \
    " m={}\n" \
    " def rn(x):return m.setdefault(x,\"p%d\"%len(m))\n" \
    " class R(ast.NodeTransformer):\n" \
    "  def visit_arg(s,n):\n" \
    "   n.arg=rn(n.arg) if n.arg in params else n.arg;return n\n" \
    "  def visit_Name(s,n):\n" \
    "   n.id=rn(n.id) if n.id in params else n.id;return n\n" \
    " fn.name=\"F\";R().visit(fn);return ast.dump(fn)\n" \
    "try:\n" \
    " a=open(sys.argv[1]).read();b=open(sys.argv[2]).read()\n" \
    " print(\"1\" if norm(a)==norm(b) else \"0\")\n" \
    "except Exception:\n" \
    " print(\"0\")\n"

static bool py_ast_equivalent(const char *cand_body, const char *helper_body) {
    if (kb_ensure_dirs() != 0) return false;
    const char *cf = ".basi/.reuse_cand.py", *hf = ".basi/.reuse_help.py";
    FILE *a = fopen(cf, "w"); if (!a) return false;
    fwrite(cand_body, 1, strlen(cand_body), a); fclose(a);
    FILE *b = fopen(hf, "w"); if (!b) { unlink(cf); return false; }
    fwrite(helper_body, 1, strlen(helper_body), b); fclose(b);

    const char *py = getenv("BASI_REUSE_PYTHON");
    if (!py || !*py) py = "python3";
    StringBuf cmd; sb_init(&cmd);
    sb_append_str(&cmd, py);
    sb_append_str(&cmd, " -c '" PY_AST_SCRIPT "' ");
    sb_append_str(&cmd, cf); sb_append_char(&cmd, ' ');
    sb_append_str(&cmd, hf); sb_append_str(&cmd, " 2>/dev/null");
    char *cmdstr = sb_to_str(&cmd);
    char *out = run_command_timeout(cmdstr, 4096, 15, NULL);
    free(cmdstr); unlink(cf); unlink(hf);
    bool eq = out && out[0] == '1';
    free(out);
    return eq;
}

/* ── Stage-2 semantic verification: Go AST equivalence ──────────────────
 * The verifier for Go (whose gc toolchain emits no LLVM IR): parse both
 * functions with go/parser, rename only their own parameters, and compare the
 * go/printer-canonicalized source. Same soundness as the IR and Python paths —
 * a renamed clone matches, a constant twin does not. Requires `go` on PATH;
 * absent → the run fails → refuse (safe, detect-only). The helper program is
 * written to a temp file and run with `go run`; the two function sources are
 * passed as NON-.go arguments so `go run` treats them as argv, not sources. */
#define GO_VERIFY_SRC \
    "package main\n" \
    "\n" \
    "import (\n" \
    "	\"bytes\"\n" \
    "	\"fmt\"\n" \
    "	\"go/ast\"\n" \
    "	\"go/parser\"\n" \
    "	\"go/printer\"\n" \
    "	\"go/token\"\n" \
    "	\"os\"\n" \
    ")\n" \
    "\n" \
    "func norm(path string) string {\n" \
    "	src, err := os.ReadFile(path)\n" \
    "	if err != nil {\n" \
    "		return \"\"\n" \
    "	}\n" \
    "	fset := token.NewFileSet()\n" \
    "	f, err := parser.ParseFile(fset, \"\", \"package p\\n\"+string(src), 0)\n" \
    "	if err != nil || len(f.Decls) == 0 {\n" \
    "		return \"\"\n" \
    "	}\n" \
    "	fn, ok := f.Decls[0].(*ast.FuncDecl)\n" \
    "	if !ok {\n" \
    "		return \"\"\n" \
    "	}\n" \
    "	m := map[string]string{}\n" \
    "	add := func(name string) {\n" \
    "		if _, seen := m[name]; !seen && name != \"_\" {\n" \
    "			m[name] = fmt.Sprintf(\"p%d\", len(m))\n" \
    "		}\n" \
    "	}\n" \
    "	if fn.Recv != nil {\n" \
    "		for _, fld := range fn.Recv.List {\n" \
    "			for _, nm := range fld.Names {\n" \
    "				add(nm.Name)\n" \
    "			}\n" \
    "		}\n" \
    "	}\n" \
    "	if fn.Type.Params != nil {\n" \
    "		for _, fld := range fn.Type.Params.List {\n" \
    "			for _, nm := range fld.Names {\n" \
    "				add(nm.Name)\n" \
    "			}\n" \
    "		}\n" \
    "	}\n" \
    "	skip := map[*ast.Ident]bool{}\n" \
    "	ast.Inspect(fn, func(n ast.Node) bool {\n" \
    "		if s, ok := n.(*ast.SelectorExpr); ok {\n" \
    "			skip[s.Sel] = true\n" \
    "		}\n" \
    "		return true\n" \
    "	})\n" \
    "	fn.Name.Name = \"F\"\n" \
    "	ast.Inspect(fn, func(n ast.Node) bool {\n" \
    "		if id, ok := n.(*ast.Ident); ok && !skip[id] {\n" \
    "			if r, ok := m[id.Name]; ok {\n" \
    "				id.Name = r\n" \
    "			}\n" \
    "		}\n" \
    "		return true\n" \
    "	})\n" \
    "	var buf bytes.Buffer\n" \
    "	printer.Fprint(&buf, fset, fn)\n" \
    "	return buf.String()\n" \
    "}\n" \
    "\n" \
    "func main() {\n" \
    "	a := norm(os.Args[1])\n" \
    "	b := norm(os.Args[2])\n" \
    "	if a != \"\" && a == b {\n" \
    "		fmt.Print(\"1\")\n" \
    "	} else {\n" \
    "		fmt.Print(\"0\")\n" \
    "	}\n" \
    "}\n"

static bool go_ast_equivalent(const char *cand_body, const char *helper_body) {
    if (kb_ensure_dirs() != 0) return false;
    /* The `go` tool ignores source files whose names start with '.', so the
     * verifier program must not be dot-prefixed (the .gosrc args are read as
     * plain files, not compiled, so their names are unconstrained). */
    const char *prog = ".basi/reuse_goverify.go";
    const char *cf = ".basi/.reuse_cand.gosrc", *hf = ".basi/.reuse_help.gosrc";
    FILE *p = fopen(prog, "w"); if (!p) return false;
    fwrite(GO_VERIFY_SRC, 1, strlen(GO_VERIFY_SRC), p); fclose(p);
    FILE *a = fopen(cf, "w"); if (!a) { unlink(prog); return false; }
    fwrite(cand_body, 1, strlen(cand_body), a); fclose(a);
    FILE *b = fopen(hf, "w"); if (!b) { unlink(prog); unlink(cf); return false; }
    fwrite(helper_body, 1, strlen(helper_body), b); fclose(b);

    const char *go = getenv("BASI_REUSE_GO");
    if (!go || !*go) go = "go";
    StringBuf cmd; sb_init(&cmd);
    sb_append_str(&cmd, go);
    sb_append_str(&cmd, " run ");
    sb_append_str(&cmd, prog); sb_append_char(&cmd, ' ');
    sb_append_str(&cmd, cf);   sb_append_char(&cmd, ' ');
    sb_append_str(&cmd, hf);   sb_append_str(&cmd, " 2>/dev/null");
    char *cmdstr = sb_to_str(&cmd);
    char *out = run_command_timeout(cmdstr, 4096, 30, NULL);
    free(cmdstr);
    unlink(prog); unlink(cf); unlink(hf);
    bool eq = out && out[0] == '1';
    free(out);
    return eq;
}

/* ── Stage-2 semantic verification: TypeScript AST equivalence ──────────
 * TypeScript has no LLVM IR and no zero-dep parser, but `deno` runs TS natively
 * and can pull the canonical TypeScript compiler on demand (npm:typescript,
 * cached in deno's own store — not vendored into the repo). The script parses
 * each function with the real TS parser, renames the function's own name +
 * params + locals, and compares the printed AST — a renamed clone matches, a
 * changed constant/operator does not. Requires `deno` on PATH (+ network on the
 * first run to fetch typescript); absent → run fails → refuse (safe). A bare
 * class-method body is retried wrapped in a dummy class so methods parse too. */
#define TS_VERIFY_SRC \
    "import ts from \"npm:typescript@5\";\n" \
    "function normOne(src: string): string {\n" \
    "  const sf = ts.createSourceFile(\"f.ts\", src, ts.ScriptTarget.Latest, true);\n" \
    "  let fn: any;\n" \
    "  const find = (n: ts.Node) => {\n" \
    "    if (!fn && (ts.isFunctionDeclaration(n) || ts.isMethodDeclaration(n) ||\n" \
    "                ts.isFunctionExpression(n) || ts.isArrowFunction(n))) fn = n;\n" \
    "    ts.forEachChild(n, find);\n" \
    "  };\n" \
    "  ts.forEachChild(sf, find);\n" \
    "  if (!fn) return \"\";\n" \
    "  const m = new Map<string, string>();\n" \
    "  const bind = (s: string) => { if (!m.has(s)) m.set(s, \"v\" + m.size); };\n" \
    "  if (fn.name && ts.isIdentifier(fn.name)) bind(fn.name.text);\n" \
    "  for (const p of fn.parameters) if (ts.isIdentifier(p.name)) bind(p.name.text);\n" \
    "  const collect = (n: ts.Node) => {\n" \
    "    if (ts.isVariableDeclaration(n) && ts.isIdentifier(n.name)) bind(n.name.text);\n" \
    "    ts.forEachChild(n, collect);\n" \
    "  };\n" \
    "  if (fn.body) collect(fn.body);\n" \
    "  const f = ts.factory;\n" \
    "  const tr = (ctx: ts.TransformationContext) => {\n" \
    "    const visit = (n: ts.Node): ts.Node => {\n" \
    "      if (ts.isIdentifier(n) && m.has(n.text)) {\n" \
    "        const par = n.parent;\n" \
    "        if (par && ts.isPropertyAccessExpression(par) && par.name === n) return n;\n" \
    "        return f.createIdentifier(m.get(n.text)!);\n" \
    "      }\n" \
    "      return ts.visitEachChild(n, visit, ctx);\n" \
    "    };\n" \
    "    return (node: ts.Node) => ts.visitNode(node, visit);\n" \
    "  };\n" \
    "  const t = ts.transform(fn, [tr]).transformed[0];\n" \
    "  return ts.createPrinter({ removeComments: true }).printNode(ts.EmitHint.Unspecified, t, sf);\n" \
    "}\n" \
    "function norm(src: string): string { let r = normOne(src); if (!r) r = normOne(\"class __W {\\n\" + src + \"\\n}\"); return r; }\n" \
    "try {\n" \
    "  const a = norm(Deno.readTextFileSync(Deno.args[0]));\n" \
    "  const b = norm(Deno.readTextFileSync(Deno.args[1]));\n" \
    "  console.log(a !== \"\" && a === b ? \"1\" : \"0\");\n" \
    "} catch { console.log(\"0\"); }\n"

static bool ts_ast_equivalent(const char *cand_body, const char *helper_body) {
    if (kb_ensure_dirs() != 0) return false;
    const char *prog = ".basi/reuse_tsnorm.ts";
    const char *cf = ".basi/.reuse_cand.ts", *hf = ".basi/.reuse_help.ts";
    FILE *p = fopen(prog, "w"); if (!p) return false;
    fwrite(TS_VERIFY_SRC, 1, strlen(TS_VERIFY_SRC), p); fclose(p);
    FILE *a = fopen(cf, "w"); if (!a) { unlink(prog); return false; }
    fwrite(cand_body, 1, strlen(cand_body), a); fclose(a);
    FILE *b = fopen(hf, "w"); if (!b) { unlink(prog); unlink(cf); return false; }
    fwrite(helper_body, 1, strlen(helper_body), b); fclose(b);

    const char *deno = getenv("BASI_REUSE_DENO");
    if (!deno || !*deno) deno = "deno";
    StringBuf cmd; sb_init(&cmd);
    sb_append_str(&cmd, deno);
    sb_append_str(&cmd, " run --quiet --allow-read --allow-env ");
    sb_append_str(&cmd, prog); sb_append_char(&cmd, ' ');
    sb_append_str(&cmd, cf);   sb_append_char(&cmd, ' ');
    sb_append_str(&cmd, hf);   sb_append_str(&cmd, " 2>/dev/null");
    char *cmdstr = sb_to_str(&cmd);
    char *out = run_command_timeout(cmdstr, 4096, 40, NULL);   /* first run may fetch typescript */
    free(cmdstr);
    unlink(prog); unlink(cf); unlink(hf);
    bool eq = out && out[0] == '1';
    free(out);
    return eq;
}

/* Dispatch Stage-2 verification by the candidate's language: compiled → LLVM IR
 * equivalence, dynamic → parser AST equivalence, otherwise refuse (we can only
 * auto-substitute what a tool can certify equivalent). */
static bool verify_equivalent(const LangBackend *be, const char *replace,
                              const FuncDef *cand, const char *cand_path,
                              const SymEntry *helper) {
    if (be->verify == VERIFY_IR)
        return ir_equivalent(replace, cand->name, cand_path, helper->file, helper->name);
    if (be->verify == VERIFY_AST && be->ast_verify)
        return be->ast_verify(cand->body, helper->body);
    return false;   /* VERIFY_NONE or no verifier → cannot certify → refuse */
}

/* Is `s` a legal identifier (used to gate Python module-name injection)? */
static bool is_ident(const char *s) {
    if (!s || !*s || isdigit((unsigned char)*s)) return false;
    for (const char *p = s; *p; p++)
        if (!(isalnum((unsigned char)*p) || *p == '_')) return false;
    return true;
}

/* Leading-whitespace width of the first line of `body` (0 for a top-level def). */
static size_t leading_indent(const char *body) {
    size_t i = 0;
    while (body[i] == ' ' || body[i] == '\t') i++;
    return i;
}

/* Byte length of `body`'s signature header — up to and including the opening
 * brace (brace langs) or the def-line colon (Python). 0 if not found. */
static size_t header_len(const FuncDef *f, const LangBackend *be) {
    if (be->brace_lang) {
        const char *br = strchr(f->body, '{');
        return br ? (size_t)(br - f->body) : 0;
    }
    const char *op = strchr(f->body, '(');
    if (!op) return 0;
    int depth = 0; const char *p = op;
    for (; *p; p++) {
        if (*p == '(' || *p == '[' || *p == '{') depth++;
        else if (*p == ')' || *p == ']' || *p == '}') { if (--depth == 0) { p++; break; } }
    }
    while (*p && *p != ':') p++;              /* skip an optional -> annotation */
    return *p == ':' ? (size_t)(p - f->body + 1) : 0;
}

/* Rebuild the whole candidate function so its body is a single call to the
 * helper, preserving the original signature line verbatim. malloc'd/NULL. */
static char *build_reuse_function(const FuncDef *f, const LangBackend *be,
                                  const char *helper, char **params, int ca) {
    size_t hl = header_len(f, be);
    if (hl == 0) return NULL;
    StringBuf b; sb_init(&b);
    sb_append(&b, f->body, hl);               /* original signature, verbatim */
    if (be->brace_lang) {
        sb_append_str(&b, "{\n    ");
        if (!sig_returns_void(f->sig)) sb_append_str(&b, "return ");
    } else {
        size_t ind = leading_indent(f->body);
        sb_append_char(&b, '\n');
        for (size_t i = 0; i < ind + 4; i++) sb_append_char(&b, ' ');
        sb_append_str(&b, "return ");
    }
    sb_append_str(&b, helper);
    sb_append_char(&b, '(');
    for (int i = 0; i < ca; i++) { if (i) sb_append_str(&b, ", "); sb_append_str(&b, params[i]); }
    sb_append_char(&b, ')');
    if (!be->brace_lang)          sb_append_char(&b, '\n');       /* Python */
    else if (be->id == LANG_GO)   sb_append_str(&b, "\n}");       /* Go: no statement ; */
    else                          sb_append_str(&b, ";\n}");      /* C / C++ */
    return sb_to_str(&b);
}

/* Directory portion of a path ("" for a bare filename). */
static void path_dir(const char *p, char *buf, size_t n) {
    const char *slash = strrchr(p, '/');
    size_t len = slash ? (size_t)(slash - p) : 0;
    if (len >= n) len = n - 1;
    memcpy(buf, p, len); buf[len] = '\0';
}

/* Injected line that makes the helper reachable from the rewritten file, plus a
 * guard that the rewrite would actually resolve. Returns a malloc'd line to
 * prepend (possibly ""), or NULL to refuse the rewrite. */
static char *reachability_line(const LangBackend *be, const char *cand_path,
                               const char *helper_file, const char *helper_name) {
    if (be->verify == VERIFY_IR) {
        /* C / C++: the helper must be declared in a sibling header. */
        char *hdr = header_for(helper_file);
        if (!hdr || !header_declares(hdr, helper_name)) { free(hdr); return NULL; }
        const char *slash = strrchr(hdr, '/');
        const char *hbase = slash ? slash + 1 : hdr;
        char inc[300]; snprintf(inc, sizeof(inc), "#include \"%s\"", hbase);
        free(hdr);
        return strdup(inc);
    }
    if (be->import_fmt) {
        /* Python: import the helper from its module (file stem). */
        const char *slash = strrchr(helper_file, '/');
        const char *base = slash ? slash + 1 : helper_file;
        const char *dot = strrchr(base, '.');
        size_t stemlen = dot ? (size_t)(dot - base) : strlen(base);
        char stem[128];
        if (stemlen == 0 || stemlen >= sizeof(stem)) return NULL;
        memcpy(stem, base, stemlen); stem[stemlen] = '\0';
        if (!is_ident(stem)) return NULL;
        char imp[300]; snprintf(imp, sizeof(imp), be->import_fmt, stem, helper_name);
        return strdup(imp);
    }
    if (be->id == LANG_GO) {
        /* Go: a same-package (same-directory) helper is visible with no import;
         * a cross-package one would need an import path we cannot infer → refuse. */
        char cd[1024], hd[1024];
        path_dir(cand_path, cd, sizeof(cd));
        path_dir(helper_file, hd, sizeof(hd));
        return strcmp(cd, hd) == 0 ? strdup("") : NULL;
    }
    return NULL;   /* no reachability mechanism → refuse */
}

char *reuse_gate_autofix(const char *path, const char *replace, const char *search) {
    if (!reuse_gate_enabled() || !env_flag("BASI_REUSE_AUTOFIX")) return NULL;
    if (!replace || !*replace) return NULL;
    if (!ensure_store()) return NULL;

    int nf = 0;
    FuncDef *fs = extract_candidate(path, replace, &nf);
    if (nf == 0) { free(fs); return NULL; }

    bool dbg = env_flag("BASI_REUSE_DEBUG");
    char *result = NULL;   /* built lazily from `replace` once a fix applies */
    int fixes = 0;

    for (int k = 0; k < nf; k++) {
        FuncDef *f = &fs[k];
        if (strlen(f->body) < MIN_BODY) continue;
        if (name_in_text(search, f->name)) continue;

        const LangBackend *be = backend_by_id(f->lang);
        if (!be) continue;

        double score;
        int mi = sym_best_match(&g_store, f, path, &score);
        if (mi < 0 || score < AUTOFIX_TAU) continue;
        SymEntry *h = &g_store.e[mi];

        char *cparams[MAX_PARAMS], *hparams[MAX_PARAMS];
        int ca = params_for(be, f->sig, cparams, MAX_PARAMS);
        int ha = params_for(be, h->sig, hparams, MAX_PARAMS);
        for (int i = 0; i < ha; i++) free(hparams[i]);
        if (ca < 0 || ca != ha) { for (int i = 0; i < ca; i++) free(cparams[i]); continue; }

        /* Type-compatibility guard: same arity is not enough (two enum→name
         * mappers over different enums match on arity but not types). C-family
         * only (the type parser is C-syntax); other langs rely on the Stage-2
         * verifier below, which is type-aware by construction. */
        if (be->verify == VERIFY_IR && !types_compatible(f->sig, h->sig)) {
            for (int i = 0; i < ca; i++) free(cparams[i]);
            continue;
        }

        /* Stage-2 semantic guard: structural + type match still can't tell a
         * behavioural twin (anetRecvTimeout vs anetSendTimeout — same shape,
         * one differing constant) from a true clone. Only substitute when a
         * verifier (compiler IR / parser AST) certifies the two functions
         * equivalent. Skippable via BASI_REUSE_NOVERIFY for A/B, but that
         * reinstates the unsafe path. */
        if (!env_flag("BASI_REUSE_NOVERIFY") &&
            !verify_equivalent(be, replace, f, path, h)) {
            if (dbg) fprintf(stderr, "reuse-autofix: `%s` not equivalent to `%s` — refusing rewrite\n",
                             f->name, h->name);
            for (int i = 0; i < ca; i++) free(cparams[i]);
            continue;
        }

        /* Reachability guard: only rewrite if the injected call would resolve
         * (C/C++: helper's header declares it; Python: helper is an importable
         * module; Go: helper is in the same package/directory). */
        char *reach = reachability_line(be, path, h->file, h->name);
        if (!reach) { for (int i = 0; i < ca; i++) free(cparams[i]); continue; }

        char *reuse_fn = build_reuse_function(f, be, h->name, cparams, ca);
        for (int i = 0; i < ca; i++) free(cparams[i]);
        if (!reuse_fn) { free(reach); continue; }

        const char *base = result ? result : replace;
        char *rewritten = str_replace_once(base, f->body, reuse_fn);
        free(reuse_fn);
        if (rewritten) {
            if (!strstr(rewritten, reach)) {          /* prepend import/include once */
                StringBuf s; sb_init(&s);
                sb_append_str(&s, reach); sb_append_char(&s, '\n');
                sb_append_str(&s, rewritten);
                free(rewritten); rewritten = sb_to_str(&s);
            }
            free(result); result = rewritten; fixes++;
            if (dbg) fprintf(stderr, "reuse-autofix: rewrote `%s` -> call `%s` (%s:%d, score %.2f)\n",
                             f->name, h->name, h->file, h->line, score);
        }
        free(reach);
    }

    for (int k = 0; k < nf; k++) funcdef_free(&fs[k]);
    free(fs);
    return fixes ? result : (free(result), (char *)NULL);
}

/* ── Behavior-preservation guard (translation-validate the model's edit) ──
 * The autofix refuses to MERGE two non-equivalent functions; the mirror risk is
 * the model doing its own merge — "generalizing" two shapes' collision into one
 * helper that silently changes one of them. Same compiler/AST oracle, aimed at
 * the edit: for every function present both before and after the edit whose body
 * changed, ask whether its behavior is preserved. Only a CONFIDENT change (both
 * versions verify and differ) is flagged — uncertainty (won't compile, inlined
 * away) never raises a false alarm. */

int reuse_regress_enabled(void) { return env_flag("BASI_REUSE_REGRESS"); }

/* Compile flags for the two temp files, from the real target's DB entry or a
 * best-effort guess. malloc'd. */
static char *regress_iflags(const char *path) {
    char *cc = cc_lookup_flags(path);
    if (cc) return cc;
    char pd[1024];
    dir_of(path, pd, sizeof(pd));
    StringBuf g; sb_init(&g);
    sb_append_str(&g, "-I. -I"); sb_append_str(&g, pd);
    return sb_to_str(&g);
}

/* Confident behavior change for a compiled function: both temp files compile,
 * the function is found in both IRs, and the normalized IR differs. */
static bool ir_fn_changed(const char *oldf, const char *newf, const char *name, const char *iflags) {
    char *iro = compile_ir(oldf, iflags), *irn = compile_ir(newf, iflags);
    bool changed = false;
    if (iro && irn) {
        char *a = ir_extract_normalize(iro, name), *b = ir_extract_normalize(irn, name);
        if (a && b) changed = (strcmp(a, b) != 0);   /* both found → compare; else don't flag */
        free(a); free(b);
    }
    free(iro); free(irn);
    return changed;
}

char *reuse_regress_check(const char *path, const char *old_src, const char *new_src) {
    if (!reuse_regress_enabled() || !old_src || !new_src) return NULL;
    const LangBackend *be = backend_by_id(lang_of_path(path));
    if (!be) return NULL;
    if (be->verify == VERIFY_NONE) return NULL;                 /* no oracle (e.g. JS) */
    if (be->verify == VERIFY_AST && !be->ast_verify) return NULL;
    if (kb_ensure_dirs() != 0) return NULL;

    const char *ext = be->exts[0];
    char oldf[64], newf[64];
    snprintf(oldf, sizeof(oldf), ".basi/.reuse_old%s", ext);
    snprintf(newf, sizeof(newf), ".basi/.reuse_new%s", ext);
    FILE *fa = fopen(oldf, "w"); if (!fa) return NULL;
    fwrite(old_src, 1, strlen(old_src), fa); fclose(fa);
    FILE *fb = fopen(newf, "w"); if (!fb) { unlink(oldf); return NULL; }
    fwrite(new_src, 1, strlen(new_src), fb); fclose(fb);

    int no = 0, nn = 0;
    FuncDef *fo = ctags_extract_file(oldf, &no);
    FuncDef *fn = ctags_extract_file(newf, &nn);
    char *iflags = (be->verify == VERIFY_IR) ? regress_iflags(path) : NULL;
    bool dbg = env_flag("BASI_REUSE_DEBUG");

    StringBuf msg; sb_init(&msg);
    int hits = 0;
    for (int i = 0; i < nn; i++) {
        FuncDef *o = NULL;
        for (int j = 0; j < no; j++)
            if (strcmp(fo[j].name, fn[i].name) == 0) { o = &fo[j]; break; }
        if (!o) continue;                                    /* newly added, not a regression */
        if (strcmp(o->body, fn[i].body) == 0) continue;      /* text unchanged */
        if (warned_seen(fn[i].name)) continue;               /* warn-once → re-issue applies */

        bool changed = (be->verify == VERIFY_IR)
            ? ir_fn_changed(oldf, newf, fn[i].name, iflags)
            : !be->ast_verify(o->body, fn[i].body);
        if (!changed) continue;

        if (hits == 0)
            sb_append_str(&msg, "edit paused — behavior guard: this edit changes the behavior of code that already exists.\n");
        char line[512];
        snprintf(line, sizeof(line),
            "  \xe2\x80\xa2 existing function `%s` behaves differently after this edit (compiler-verified)\n",
            fn[i].name);
        sb_append_str(&msg, line);
        warned_add(fn[i].name);
        hits++;
        if (dbg) fprintf(stderr, "reuse-regress: `%s` behavior changed\n", fn[i].name);
    }

    for (int i = 0; i < no; i++) funcdef_free(&fo[i]);
    for (int i = 0; i < nn; i++) funcdef_free(&fn[i]);
    free(fo); free(fn); free(iflags);
    unlink(oldf); unlink(newf);

    if (hits == 0) { sb_free(&msg); return NULL; }
    sb_append_str(&msg,
        "\nIf these behavior changes are intended, re-issue this same edit — it will apply. "
        "If not (e.g. you consolidated shared logic and altered one existing caller), fix it so "
        "the existing behavior is preserved before re-issuing.\n");
    return sb_to_str(&msg);
}

void reuse_shutdown(void) {
    if (g_loaded) { sym_clear(&g_store); g_loaded = false; }
    for (int i = 0; i < g_warned_n; i++) free(g_warned[i]);
    free(g_warned);
    g_warned = NULL; g_warned_n = g_warned_cap = 0;
}
