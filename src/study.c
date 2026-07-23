#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <regex.h>
#include <errno.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/wait.h>

#include "util.h"
#include "kb.h"
#include "srvchat.h"
#include "memory.h"
#include "embed.h"    /* embed_init — spawn the shared embedder before forking */
#include "study.h"
#include "model.h"      /* extract_tool_call, basi_srv_suppress_grammar (grounding) */
#include "web.h"        /* execute_readfile / execute_web_search / execute_web_fetch */
#include "symbols.h"    /* execute_symbols */
#include "chat_tmpl.h"  /* basi_set_tools / basi_tools_registered */
#include "tooldefs.h"   /* basi_tool_defs — restore tool grammar after grounding */

static void cli_progress(const char *arm, int run, int of, void *ud);

/* ── Study artifact: validate, execute, decide ─────────────────────────
 *
 * Everything in this file is deterministic. The model writes the artifact
 * and reads the verdict; it never computes one. See study.h.
 */

static const char *STUDY_REQUIRED_FM_KEYS[] = {
    "slug", "status", "created", "metric", "extract", "runs", "decision_rule", NULL
};

static const char *STUDY_VALID_STATUSES[] = {
    "proposed", "running", "complete", "abandoned", NULL
};

static const char *STUDY_SECTIONS[] = {
    "Question", "Hypothesis", "Experiment", "Results", "Verdict", NULL
};

const char *study_verdict_name(StudyVerdict v) {
    switch (v) {
        case VERDICT_SUPPORTED:    return "SUPPORTED";
        case VERDICT_REFUTED:      return "REFUTED";
        default:                   return "INCONCLUSIVE";
    }
}

/* ── Small text helpers (mirrors plan.c) ───────────────────────────────── */

static bool st_icaseq(const char *a, size_t alen, const char *b) {
    size_t blen = strlen(b);
    if (alen != blen) return false;
    for (size_t i = 0; i < alen; i++) {
        char ca = a[i], cb = b[i];
        if (ca >= 'A' && ca <= 'Z') ca += 32;
        if (cb >= 'A' && cb <= 'Z') cb += 32;
        if (ca != cb) return false;
    }
    return true;
}

/* Locate the body under a "## <name>" heading. Returns start pointer and sets
 * *out_len to the bytes up to the next "## " heading (or end of text). */
static const char *study_section(const char *body, size_t body_len,
                                 const char *name, size_t *out_len) {
    const char *p = body, *end = body + body_len;
    while (p < end) {
        const char *le = p;
        while (le < end && *le != '\n') le++;
        if (le > p + 2 && p[0] == '#' && p[1] == '#' && p[2] == ' ') {
            const char *h = p + 3;
            while (h < le && (*h == ' ' || *h == '\t')) h++;
            const char *he = le;
            while (he > h && (he[-1] == ' ' || he[-1] == '\t' ||
                              he[-1] == '#' || he[-1] == '\r')) he--;
            if (st_icaseq(h, he - h, name)) {
                const char *bstart = (le < end) ? le + 1 : end;
                const char *q = bstart;
                while (q < end) {
                    const char *qe = q;
                    while (qe < end && *qe != '\n') qe++;
                    if (qe > q + 2 && q[0] == '#' && q[1] == '#' && q[2] == ' ') break;
                    q = (qe < end) ? qe + 1 : end;
                }
                if (out_len) *out_len = (size_t)(q - bstart);
                return bstart;
            }
        }
        p = (le < end) ? le + 1 : end;
    }
    if (out_len) *out_len = 0;
    return NULL;
}

static bool study_has_h2(const char *body, size_t body_len, const char *name) {
    return study_section(body, body_len, name, NULL) != NULL;
}

static char *trim_dup(const char *s, size_t n) {
    while (n && (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n')) { s++; n--; }
    while (n && (s[n-1] == ' ' || s[n-1] == '\t' || s[n-1] == '\r' || s[n-1] == '\n')) n--;
    char *out = malloc(n + 1);
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

/* ── Arm extraction ────────────────────────────────────────────────────
 * Arms are fenced blocks in ## Experiment whose info string is "arm <name>":
 *
 *   ```arm A
 *   ./bench.sh --baseline
 *   ```
 *
 * A fence keeps multi-line commands unambiguous, which a YAML scalar would
 * not. Returns the number of arms found, or -1 with *err set (malloc'd).
 */
static int study_parse_arms(const char *body, size_t body_len,
                            StudyArm *arms, int max_arms, char **err) {
    if (err) *err = NULL;
    size_t sec_len = 0;
    const char *sec = study_section(body, body_len, "Experiment", &sec_len);
    if (!sec) return 0;

    int n = 0;
    const char *p = sec, *end = sec + sec_len;
    while (p < end) {
        const char *le = p;
        while (le < end && *le != '\n') le++;

        if (le - p >= 3 && strncmp(p, "```", 3) == 0) {
            const char *info = p + 3;
            while (info < le && (*info == ' ' || *info == '\t')) info++;
            if (le - info >= 3 && strncasecmp(info, "arm", 3) == 0 &&
                (info[3] == ' ' || info[3] == '\t')) {
                const char *nm = info + 4;
                while (nm < le && (*nm == ' ' || *nm == '\t')) nm++;
                const char *nme = le;
                while (nme > nm && (nme[-1] == ' ' || nme[-1] == '\t' ||
                                    nme[-1] == '\r')) nme--;
                if (nme == nm) {
                    if (err) *err = strdup("- an ```arm fence has no name (use ```arm A)\n");
                    return -1;
                }
                if (n >= max_arms) {
                    char *m = malloc(96);
                    snprintf(m, 96, "- too many arms (max %d)\n", max_arms);
                    if (err) *err = m;
                    return -1;
                }

                /* Collect until a closing fence. */
                const char *cstart = (le < end) ? le + 1 : end;
                const char *q = cstart;
                const char *cend = NULL;
                while (q < end) {
                    const char *qe = q;
                    while (qe < end && *qe != '\n') qe++;
                    const char *t = q;
                    while (t < qe && (*t == ' ' || *t == '\t')) t++;
                    if (qe - t >= 3 && strncmp(t, "```", 3) == 0) { cend = q; break; }
                    q = (qe < end) ? qe + 1 : end;
                }
                if (!cend) {
                    if (err) *err = strdup("- an ```arm fence is never closed\n");
                    return -1;
                }

                arms[n].name    = trim_dup(nm, (size_t)(nme - nm));
                arms[n].command = trim_dup(cstart, (size_t)(cend - cstart));
                arms[n].nruns   = 0;
                if (!*arms[n].command) {
                    char *m = malloc(128);
                    snprintf(m, 128, "- arm '%s' has an empty command\n", arms[n].name);
                    free(arms[n].name); free(arms[n].command);
                    if (err) *err = m;
                    return -1;
                }
                for (int i = 0; i < n; i++) {
                    if (strcmp(arms[i].name, arms[n].name) == 0) {
                        char *m = malloc(128);
                        snprintf(m, 128, "- duplicate arm name '%s'\n", arms[n].name);
                        free(arms[n].name); free(arms[n].command);
                        if (err) *err = m;
                        return -1;
                    }
                    /* Two arms running the SAME command is not an experiment, it
                     * is one measurement taken twice, and the difference between
                     * them is pure noise. Observed for real: a model edited the
                     * file under test, pointed both arms at the same build-and-run
                     * script, and reported the 1.3% gap between two runs of the
                     * IDENTICAL binary as a win — while the change it had actually
                     * made was worth 11.9x. Nothing downstream can detect this, so
                     * it has to die here. */
                    if (strcmp(arms[i].command, arms[n].command) == 0) {
                        char *m = malloc(512);
                        snprintf(m, 512,
                            "- arms '%s' and '%s' run the SAME command, so they measure the same "
                            "thing and their difference is only noise. Each arm must run a "
                            "DIFFERENT configuration. If you are comparing a change against a "
                            "baseline, the arms must select between them (a flag, an env var, or "
                            "separate build outputs) — editing the file in place leaves no "
                            "baseline to measure.\n",
                            arms[i].name, arms[n].name);
                        free(arms[n].name); free(arms[n].command);
                        if (err) *err = m;
                        return -1;
                    }
                }
                n++;
                /* Resume after the closing fence line. */
                const char *after = cend;
                while (after < end && *after != '\n') after++;
                p = (after < end) ? after + 1 : end;
                continue;
            }
        }
        p = (le < end) ? le + 1 : end;
    }
    return n;
}

/* ── Decision-rule grammar ─────────────────────────────────────────────
 *
 *   rule     := operand OP operand [ "and" "p" "<" number ]
 *   operand  := agg "(" armname ")" [ ("+"|"-") number ]  |  number
 *   agg      := mean | median | min | max | sd
 *   OP       := "<" | ">" | "<=" | ">="
 *
 * Deliberately tiny and total: every accepted string has exactly one meaning,
 * and anything else is rejected at validation time rather than at 3am mid-run.
 */

typedef struct {
    bool   is_const;
    double constant;
    char   agg[12];
    char   arm[64];
    double offset;
} ROperand;

typedef struct {
    ROperand lhs, rhs;
    char     op[3];
    bool     has_p;
    double   p_threshold;
} Rule;

static void rskip(const char **p) { while (**p == ' ' || **p == '\t') (*p)++; }

static bool rparse_number(const char **p, double *out) {
    char *end = NULL;
    double v = strtod(*p, &end);
    if (end == *p) return false;
    *p = end;
    *out = v;
    return true;
}

static bool ragg_valid(const char *a) {
    return strcmp(a, "mean") == 0 || strcmp(a, "median") == 0 ||
           strcmp(a, "min")  == 0 || strcmp(a, "max")    == 0 ||
           strcmp(a, "sd")   == 0;
}

static bool rparse_operand(const char **p, ROperand *o, char *err, size_t errsz) {
    rskip(p);
    memset(o, 0, sizeof(*o));
    if (**p == '-' || **p == '.' || (**p >= '0' && **p <= '9')) {
        if (!rparse_number(p, &o->constant)) {
            snprintf(err, errsz, "expected a number near '%.20s'", *p);
            return false;
        }
        o->is_const = true;
        return true;
    }
    size_t i = 0;
    while (((**p >= 'a' && **p <= 'z') || (**p >= 'A' && **p <= 'Z')) &&
           i + 1 < sizeof(o->agg)) {
        o->agg[i++] = *(*p)++;
    }
    o->agg[i] = '\0';
    for (char *c = o->agg; *c; c++) if (*c >= 'A' && *c <= 'Z') *c += 32;
    if (i == 0) {
        snprintf(err, errsz,
                 "expected an aggregate or a number near '%.20s'", *p);
        return false;
    }
    if (!ragg_valid(o->agg)) {
        snprintf(err, errsz,
                 "unknown aggregate '%s'; use mean | median | min | max | sd", o->agg);
        return false;
    }
    rskip(p);
    if (**p != '(') { snprintf(err, errsz, "expected '(' after %s", o->agg); return false; }
    (*p)++;
    rskip(p);
    i = 0;
    while (**p && **p != ')' && i + 1 < sizeof(o->arm)) o->arm[i++] = *(*p)++;
    o->arm[i] = '\0';
    if (**p != ')') { snprintf(err, errsz, "expected ')' after arm name"); return false; }
    (*p)++;
    /* trim arm name */
    size_t alen = strlen(o->arm);
    while (alen && (o->arm[alen-1] == ' ' || o->arm[alen-1] == '\t')) o->arm[--alen] = '\0';
    if (!alen) { snprintf(err, errsz, "empty arm name in %s()", o->agg); return false; }

    rskip(p);
    if (**p == '+' || **p == '-') {
        int sign = (**p == '-') ? -1 : 1;
        (*p)++;
        rskip(p);
        double v;
        if (!rparse_number(p, &v)) {
            snprintf(err, errsz, "expected a number after '%c'", sign < 0 ? '-' : '+');
            return false;
        }
        o->offset = sign * v;
    }
    return true;
}

static bool rule_parse(const char *src, Rule *r, char *err, size_t errsz) {
    memset(r, 0, sizeof(*r));
    const char *p = src;
    if (!rparse_operand(&p, &r->lhs, err, errsz)) return false;
    rskip(&p);
    if (p[0] == '<' && p[1] == '=')      { strcpy(r->op, "<="); p += 2; }
    else if (p[0] == '>' && p[1] == '=') { strcpy(r->op, ">="); p += 2; }
    else if (p[0] == '<')                { strcpy(r->op, "<");  p += 1; }
    else if (p[0] == '>')                { strcpy(r->op, ">");  p += 1; }
    else {
        snprintf(err, errsz, "expected a comparison operator (< > <= >=) near '%.20s'", p);
        return false;
    }
    if (!rparse_operand(&p, &r->rhs, err, errsz)) return false;

    rskip(&p);
    if (*p) {
        if (strncasecmp(p, "and", 3) != 0) {
            snprintf(err, errsz, "unexpected trailing text '%.24s' (only 'and p < X' may follow)", p);
            return false;
        }
        p += 3;
        rskip(&p);
        if (*p != 'p' && *p != 'P') {
            snprintf(err, errsz, "expected 'p' after 'and'");
            return false;
        }
        p++;
        rskip(&p);
        if (*p == '<' && p[1] == '=') p += 2;
        else if (*p == '<')           p += 1;
        else { snprintf(err, errsz, "expected '<' after 'p'"); return false; }
        rskip(&p);
        if (!rparse_number(&p, &r->p_threshold)) {
            snprintf(err, errsz, "expected a number after 'p <'");
            return false;
        }
        if (r->p_threshold <= 0.0 || r->p_threshold >= 1.0) {
            snprintf(err, errsz, "p threshold must be strictly between 0 and 1");
            return false;
        }
        r->has_p = true;
        rskip(&p);
        if (*p) {
            snprintf(err, errsz, "unexpected trailing text '%.24s'", p);
            return false;
        }
    }
    if (r->has_p && (r->lhs.is_const || r->rhs.is_const)) {
        snprintf(err, errsz,
                 "'and p < X' needs two arms to compare; one side is a constant");
        return false;
    }
    return true;
}

/* ── Statistics ────────────────────────────────────────────────────────── */

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Collect the values of an arm's OK runs. Returns count. */
static int arm_values(const StudyArm *arm, double *out, int max) {
    int n = 0;
    for (int i = 0; i < arm->nruns && n < max; i++) {
        if (arm->runs[i].status == RUN_OK) out[n++] = arm->runs[i].value;
    }
    return n;
}

StudyStats study_arm_stats(const StudyArm *arm) {
    StudyStats s;
    memset(&s, 0, sizeof(s));
    double v[STUDY_MAX_RUNS];
    int n = arm_values(arm, v, STUDY_MAX_RUNS);
    s.n = n;
    if (n == 0) return s;

    double sum = 0;
    for (int i = 0; i < n; i++) sum += v[i];
    s.mean = sum / n;

    double ss = 0;
    for (int i = 0; i < n; i++) ss += (v[i] - s.mean) * (v[i] - s.mean);
    s.sd = (n > 1) ? sqrt(ss / (n - 1)) : 0.0;

    qsort(v, n, sizeof(double), cmp_double);
    s.min = v[0];
    s.max = v[n - 1];
    s.median = (n % 2) ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
    return s;
}

/* Exact two-sided Mann-Whitney p via the standard counting recurrence
 *   c(n,m,u) = c(n-1,m,u-m) + c(n,m-1,u)
 * Valid only without ties. */
static double mwu_exact_p(int n1, int n2, double U) {
    int maxU = n1 * n2;
    size_t width = (size_t)maxU + 1;
    /* prev[j] holds c(i-1, j, ·); cur[j] holds c(i, j, ·) */
    double *cur = calloc((size_t)(n2 + 1) * width, sizeof(double));
    double *prev = calloc((size_t)(n2 + 1) * width, sizeof(double));
    if (!cur || !prev) { free(cur); free(prev); return -1.0; }

    /* i = 0: c(0, j, 0) = 1 for all j */
    for (int j = 0; j <= n2; j++) prev[(size_t)j * width + 0] = 1.0;

    for (int i = 1; i <= n1; i++) {
        for (int j = 0; j <= n2; j++) {
            double *row = cur + (size_t)j * width;
            memset(row, 0, width * sizeof(double));
            if (j == 0) { row[0] = 1.0; continue; }
            const double *up   = prev + (size_t)j * width;        /* c(i-1, j, u-j) */
            const double *left = cur  + (size_t)(j - 1) * width;  /* c(i, j-1, u)   */
            for (int u = 0; u <= maxU; u++) {
                double val = left[u];
                if (u - j >= 0) val += up[u - j];
                row[u] = val;
            }
        }
        double *tmp = prev; prev = cur; cur = tmp;
    }

    const double *dist = prev + (size_t)n2 * width;
    double total = 0;
    for (int u = 0; u <= maxU; u++) total += dist[u];
    if (total <= 0) { free(cur); free(prev); return -1.0; }

    int Ui = (int)(U + 0.5);
    double lo = 0, hi = 0;
    for (int u = 0; u <= maxU; u++) {
        if (u <= Ui) lo += dist[u];
        if (u >= Ui) hi += dist[u];
    }
    free(cur);
    free(prev);
    double p = 2.0 * ((lo < hi) ? lo : hi) / total;
    return (p > 1.0) ? 1.0 : p;
}

double study_mannwhitney_p(const double *a, int na, const double *b, int nb) {
    if (na < 2 || nb < 2) return -1.0;
    int N = na + nb;

    /* Combined sample with midranks for ties. */
    typedef struct { double v; int grp; double rank; } Item;
    Item *items = malloc((size_t)N * sizeof(Item));
    if (!items) return -1.0;
    for (int i = 0; i < na; i++) { items[i].v = a[i]; items[i].grp = 0; }
    for (int i = 0; i < nb; i++) { items[na + i].v = b[i]; items[na + i].grp = 1; }
    for (int i = 1; i < N; i++) {              /* insertion sort by value */
        Item key = items[i];
        int j = i - 1;
        while (j >= 0 && items[j].v > key.v) { items[j + 1] = items[j]; j--; }
        items[j + 1] = key;
    }

    bool has_ties = false;
    double tie_sum = 0;                        /* sum of (t^3 - t) */
    for (int i = 0; i < N; ) {
        int j = i;
        while (j + 1 < N && items[j + 1].v == items[i].v) j++;
        double rank = (i + j) / 2.0 + 1.0;     /* midrank, 1-based */
        for (int k = i; k <= j; k++) items[k].rank = rank;
        int t = j - i + 1;
        if (t > 1) { has_ties = true; tie_sum += (double)t * t * t - t; }
        i = j + 1;
    }

    double R1 = 0;
    for (int i = 0; i < N; i++) if (items[i].grp == 0) R1 += items[i].rank;
    free(items);

    double U1 = R1 - (double)na * (na + 1) / 2.0;
    double U2 = (double)na * nb - U1;
    double U = (U1 < U2) ? U1 : U2;

    if (!has_ties && na <= 20 && nb <= 20) {
        double p = mwu_exact_p(na, nb, U);
        if (p >= 0) return p;
    }

    /* Normal approximation with tie correction and continuity correction. */
    double mu = (double)na * nb / 2.0;
    double var = ((double)na * nb / 12.0) *
                 ((double)(N + 1) - tie_sum / ((double)N * (N - 1)));
    if (var <= 0) return 1.0;
    double z = (fabs(U - mu) - 0.5) / sqrt(var);
    if (z < 0) z = 0;
    double p = erfc(z / sqrt(2.0));            /* two-sided */
    return (p > 1.0) ? 1.0 : p;
}

/* ── Validation ────────────────────────────────────────────────────────── */

char *validate_study_artifact(const char *content, size_t content_len) {
    StringBuf errs;
    sb_init(&errs);

    KbFrontmatter fm;
    int rc = kb_parse_frontmatter(content, content_len, &fm);
    if (rc < 0) {
        sb_append_str(&errs, "- malformed YAML frontmatter (missing closing ---)\n");
        return sb_to_str(&errs);
    }
    if (fm.body_offset == 0) {
        sb_append_str(&errs, "- missing frontmatter block (must start with ---)\n");
        kb_fm_free(&fm);
        return sb_to_str(&errs);
    }

    for (const char **k = STUDY_REQUIRED_FM_KEYS; *k; k++) {
        if (!kb_fm_get(&fm, *k)) {
            sb_append_str(&errs, "- frontmatter missing required key: ");
            sb_append_str(&errs, *k);
            sb_append_char(&errs, '\n');
        }
    }

    const char *slug = kb_fm_get(&fm, "slug");
    if (slug && !kb_slug_valid(slug)) {
        sb_append_str(&errs, "- slug '");
        sb_append_str(&errs, slug);
        sb_append_str(&errs,
            "' invalid (lowercase a-z, 0-9, single hyphens; must start with a letter; max 64 chars)\n");
    }

    const char *status = kb_fm_get(&fm, "status");
    if (status) {
        bool ok = false;
        for (const char **s = STUDY_VALID_STATUSES; *s; s++)
            if (strcmp(*s, status) == 0) { ok = true; break; }
        if (!ok) {
            sb_append_str(&errs, "- status '");
            sb_append_str(&errs, status);
            sb_append_str(&errs, "' invalid; must be one of: proposed | running | complete | abandoned\n");
        }
    }

    /* runs: the noise floor measured on this machine is large enough that a
     * 1-run or 2-run comparison is not evidence. Demand 5, or a written
     * justification the author has to look at while typing it. */
    const char *runs_s = kb_fm_get(&fm, "runs");
    long runs = 0;
    if (runs_s) {
        char *end = NULL;
        runs = strtol(runs_s, &end, 10);
        if (end == runs_s || *end) {
            sb_append_str(&errs, "- runs must be an integer\n");
        } else if (runs < 1 || runs > STUDY_MAX_RUNS) {
            char buf[128];
            snprintf(buf, sizeof(buf), "- runs must be between 1 and %d (got %ld)\n",
                     STUDY_MAX_RUNS, runs);
            sb_append_str(&errs, buf);
        } else if (runs < STUDY_MIN_RUNS && !kb_fm_get(&fm, "low_n_justification")) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                "- runs is %ld but the measured noise floor needs >= %d per arm. "
                "Raise runs, or add a low_n_justification: <why this is still evidence> key.\n",
                runs, STUDY_MIN_RUNS);
            sb_append_str(&errs, buf);
        }
    }

    const char *timeout_s = kb_fm_get(&fm, "timeout");
    if (timeout_s) {
        char *end = NULL;
        long t = strtol(timeout_s, &end, 10);
        if (end == timeout_s || *end || t < 1) {
            sb_append_str(&errs, "- timeout must be a positive integer (seconds)\n");
        }
    }

    /* extract: must compile and must capture a group to read the number from. */
    const char *extract = kb_fm_get(&fm, "extract");
    if (extract) {
        regex_t re;
        int erc = regcomp(&re, extract, REG_EXTENDED);
        if (erc != 0) {
            char ebuf[160];
            regerror(erc, &re, ebuf, sizeof(ebuf));
            char buf[320];
            snprintf(buf, sizeof(buf), "- extract regex does not compile: %s\n", ebuf);
            sb_append_str(&errs, buf);
        } else {
            if (re.re_nsub < 1) {
                sb_append_str(&errs,
                    "- extract regex has no capture group; group 1 must capture the number "
                    "(e.g. rounds=([0-9.]+))\n");
            }
            regfree(&re);
        }
    }

    const char *body = content + fm.body_offset;
    size_t body_len = content_len - fm.body_offset;

    for (const char **sec = STUDY_SECTIONS; *sec; sec++) {
        if (!study_has_h2(body, body_len, *sec)) {
            sb_append_str(&errs, "- missing required section: ## ");
            sb_append_str(&errs, *sec);
            sb_append_char(&errs, '\n');
        }
    }

    /* Arms, and the decision rule that must reference them. */
    StudyArm arms[STUDY_MAX_ARMS];
    char *aerr = NULL;
    int narms = study_parse_arms(body, body_len, arms, STUDY_MAX_ARMS, &aerr);
    if (narms < 0) {
        sb_append_str(&errs, aerr ? aerr : "- could not parse arms\n");
        free(aerr);
        narms = 0;
    } else if (narms == 0) {
        sb_append_str(&errs,
            "- ## Experiment defines no arms; add a fenced block per arm:\n"
            "  ```arm A\n  <shell command that prints the metric>\n  ```\n");
    }

    const char *rule_src = kb_fm_get(&fm, "decision_rule");
    if (rule_src) {
        Rule r;
        char rerr[192];
        if (!rule_parse(rule_src, &r, rerr, sizeof(rerr))) {
            char buf[384];
            snprintf(buf, sizeof(buf), "- decision_rule invalid: %s\n", rerr);
            sb_append_str(&errs, buf);
        } else if (!r.has_p && !r.lhs.is_const && !r.rhs.is_const) {
            /* Comparing two arms without a significance clause lets any noise
             * gap pass as a win — measured: a 1.3% difference between two runs
             * of the SAME binary was reported as SUPPORTED. A threshold rule
             * against a constant is exempt; arm-vs-arm is not. */
            sb_append_str(&errs,
                "- decision_rule compares two arms but pre-registers no significance clause. "
                "Append 'and p < 0.05' (or your chosen threshold). Without it, a difference "
                "indistinguishable from noise counts as a win — which is how a 1.3% gap between "
                "two runs of the same binary once passed as an improvement.\n");
        } else if (narms > 0) {
            const ROperand *ops[2] = { &r.lhs, &r.rhs };
            for (int i = 0; i < 2; i++) {
                if (ops[i]->is_const) continue;
                bool found = false;
                for (int j = 0; j < narms; j++)
                    if (strcmp(arms[j].name, ops[i]->arm) == 0) { found = true; break; }
                if (!found) {
                    char buf[256];
                    snprintf(buf, sizeof(buf),
                        "- decision_rule references arm '%s', which has no ```arm block\n",
                        ops[i]->arm);
                    sb_append_str(&errs, buf);
                }
            }
        }
    }

    for (int i = 0; i < narms; i++) { free(arms[i].name); free(arms[i].command); }
    kb_fm_free(&fm);

    if (errs.len == 0) { sb_free(&errs); return NULL; }
    return sb_to_str(&errs);
}

/* ── Parse ─────────────────────────────────────────────────────────────── */

void study_free(Study *s) {
    if (!s) return;
    free(s->slug); free(s->metric); free(s->extract); free(s->decision_rule);
    for (int i = 0; i < s->narms; i++) {
        free(s->arms[i].name); free(s->arms[i].command);
        for (int j = 0; j < s->arms[i].nruns; j++) free(s->arms[i].runs[j].sample);
    }
    memset(s, 0, sizeof(*s));
}

char *study_parse(const char *content, size_t content_len, Study *out) {
    memset(out, 0, sizeof(*out));
    char *errs = validate_study_artifact(content, content_len);
    if (errs) return errs;

    KbFrontmatter fm;
    kb_parse_frontmatter(content, content_len, &fm);
    out->slug          = strdup(kb_fm_get(&fm, "slug"));
    out->metric        = strdup(kb_fm_get(&fm, "metric"));
    out->extract       = strdup(kb_fm_get(&fm, "extract"));
    out->decision_rule = strdup(kb_fm_get(&fm, "decision_rule"));
    out->runs          = (int)strtol(kb_fm_get(&fm, "runs"), NULL, 10);

    const char *t = kb_fm_get(&fm, "timeout");
    out->timeout_s = t ? (int)strtol(t, NULL, 10) : STUDY_DEFAULT_TIMEOUT;

    const char *body = content + fm.body_offset;
    size_t body_len = content_len - fm.body_offset;
    char *aerr = NULL;
    int n = study_parse_arms(body, body_len, out->arms, STUDY_MAX_ARMS, &aerr);
    kb_fm_free(&fm);
    if (n < 0) { study_free(out); return aerr ? aerr : strdup("could not parse arms\n"); }
    out->narms = n;
    return NULL;
}

/* ── Execution ─────────────────────────────────────────────────────────── */

#define RC_SENTINEL "__BASI_STUDY_RC="

/* Pull the last capture-group-1 match out of `text`. Returns true if found. */
/* Returns true if a number was extracted. `spread` (if non-NULL) receives the
 * ratio between the largest and smallest DISTINCT values the regex matched in
 * this one run — 1.0 when it matched a single quantity.
 *
 * A regex that matches several very different numbers in one invocation is not
 * identifying a metric, it is sampling whichever happened to come last.
 * Measured: `zstd -b` prints compression AND decompression throughput on the
 * same line, `([0-9.]+)\s*MB/s` matched both, and one arm was scored on
 * decompression (~2580) against another on compression (~405). The loop
 * reported SUPPORTED at p=1e-10 for a comparison between two different
 * quantities. */
static bool extract_metric(const regex_t *re, const char *text, double *out,
                           double *spread, double *out_lo, double *out_hi) {
    regmatch_t m[2];
    const char *cur = text;
    bool found = false;
    double last = 0, lo = 0, hi = 0;
    int flags = 0;
    if (spread) *spread = 1.0;
    while (*cur && regexec(re, cur, 2, m, flags) == 0) {
        if (m[1].rm_so >= 0) {
            size_t len = (size_t)(m[1].rm_eo - m[1].rm_so);
            char buf[64];
            if (len < sizeof(buf)) {
                memcpy(buf, cur + m[1].rm_so, len);
                buf[len] = '\0';
                char *end = NULL;
                double v = strtod(buf, &end);
                if (end != buf) {
                    if (!found || v < lo) lo = v;
                    if (!found || v > hi) hi = v;
                    last = v; found = true;
                }
            }
        }
        regoff_t adv = (m[0].rm_eo > m[0].rm_so) ? m[0].rm_eo : m[0].rm_so + 1;
        cur += adv;
        flags = REG_NOTBOL;
    }
    if (found) *out = last;
    if (found && spread && lo > 0) *spread = hi / lo;
    if (found && out_lo) *out_lo = lo;
    if (found && out_hi) *out_hi = hi;
    return found;
}

/* SOURCE files only. A benchmark legitimately writes scratch state, logs,
 * objects and binaries while it runs; hashing those flags every honest study as
 * contaminated (measured: it turned all four regression fixtures INCONCLUSIVE,
 * because their harness keeps a counter file). What must not change between
 * arms is the CODE under test, so only source extensions are fingerprinted. */
static bool is_source_file(const char *name) {
    const char *dot = strrchr(name, '.');
    if (!dot) return false;
    static const char *ext[] = { ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh",
                                 ".py", ".rs", ".go", ".js", ".ts", ".java", ".sh",
                                 ".cl", ".cu", ".glsl", ".comp", ".m", ".swift", NULL };
    for (const char **e = ext; *e; e++) if (strcmp(dot, *e) == 0) return true;
    return false;
}

/* Fingerprint the source tree: each source file's path, size and mtime folded
 * into one number. Cheap, and enough to notice that an arm rewrote the code
 * under test. .basi/ is skipped because the runner itself writes there. */
static void tree_hash_walk(const char *dir, unsigned long *h, int depth) {
    if (depth > 4) return;
    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;         /* skips .basi, .git, dotfiles */
        char path[2048];
        if ((size_t)snprintf(path, sizeof(path), "%s/%s", dir, e->d_name) >= sizeof(path)) continue;
        struct stat st;
        if (stat(path, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) { tree_hash_walk(path, h, depth + 1); continue; }
        if (!S_ISREG(st.st_mode)) continue;
        if (!is_source_file(e->d_name)) continue;
        for (const char *p = e->d_name; *p; p++) *h = *h * 131u + (unsigned char)*p;
        *h = *h * 1000003u + (unsigned long)st.st_size;
        *h = *h * 1000003u + (unsigned long)st.st_mtime;
    }
    closedir(d);
}

static unsigned long tree_fingerprint(void) {
    unsigned long h = 1469598103u;
    tree_hash_walk(".", &h, 0);
    return h;
}

/* Collapse a command's output into a short readable excerpt for the Results
 * section: strip CR (progress spinners overwrite with them), fold newlines and
 * runs of blanks, and cap the length.
 *
 * HEAD *AND* TAIL, because the useful part is usually at the END. Measured: four
 * unattended rounds tried to fix an extract regex against llama-bench while the
 * excerpt showed only its Vulkan device banner — the results table they needed
 * was past the cut. This is the same finding as c03f7c2, which replaced
 * head-only tool-result truncation for exactly this reason. */
static char *squash_range(const char *p, const char *end, char *o, size_t cap, size_t *np) {
    size_t n = *np;
    bool sp = false;
    for (; p < end && n < cap; p++) {
        unsigned char c = (unsigned char)*p;
        if (c == '\r' || c == '\n' || c == '\t' || c == ' ') { sp = true; continue; }
        if (c < 0x20) continue;
        if (sp && n) o[n++] = ' ';
        sp = false;
        if (n < cap) o[n++] = (char)c;
    }
    *np = n;
    return o;
}

static char *squash_output(const char *out) {
    if (!out || !*out) return strdup("(no output)");
    const size_t HEAD = 140, TAIL = 220;
    size_t len = strlen(out);
    size_t cap = HEAD + TAIL + 8;
    char *o = malloc(cap + 8);
    size_t n = 0;

    if (len <= HEAD + TAIL) {
        squash_range(out, out + len, o, cap, &n);
    } else {
        squash_range(out, out + HEAD, o, HEAD, &n);
        const char *mid = " ... ";
        for (const char *m = mid; *m && n < cap; m++) o[n++] = *m;
        squash_range(out + len - TAIL, out + len, o, cap, &n);
    }
    while (n && o[n-1] == ' ') n--;
    o[n] = '\0';
    if (!n) { free(o); return strdup("(no output)"); }
    return o;
}

/* Abbreviate a command for a summary line, keeping the TAIL.
 *
 * What distinguishes two arms is almost always at the end — the flags — while
 * the head is a long shared path. Measured: the cross-trajectory brief truncated
 * arm commands at 180 chars, which a model path consumed entirely, so every arm
 * read as identical and later trajectories could not see what earlier ones had
 * actually varied. Same head-vs-tail lesson as the failure excerpts. */
static char *abbrev_cmd(const char *cmd) {
    if (!cmd) return strdup("(none)");
    size_t len = strlen(cmd);
    const size_t HEAD = 40, TAIL = 130;
    if (len <= HEAD + TAIL) return strdup(cmd);
    char *o = malloc(HEAD + TAIL + 8);
    memcpy(o, cmd, HEAD);
    memcpy(o + HEAD, " … ", 5);                 /* U+2026, 3 bytes + 2 spaces */
    memcpy(o + HEAD + 5, cmd + len - TAIL, TAIL);
    o[HEAD + 5 + TAIL] = '\0';
    return o;
}

void study_execute(Study *s, StudyProgressFn progress, void *ud) {
    regex_t re;
    if (regcomp(&re, s->extract, REG_EXTENDED) != 0) return;  /* validated already */

    /* Interleave arms round by round (A,B,A,B,…) instead of all of A then all of
     * B, so any systematic drift over the session — GPU thermal warm-up, memory
     * state, background load — hits both arms equally and cancels in the compare.
     * All-A-then-all-B lets a warm-up bias masquerade as a real and even
     * statistically SIGNIFICANT arm difference (measured: a Q4_K kernel no-op read
     * as +2 tok/s because the baseline was benched cold and the patch warm). */
    for (int a = 0; a < s->narms; a++) s->arms[a].nruns = 0;
    for (int i = 0; i < s->runs && i < STUDY_MAX_RUNS; i++) {
        for (int a = 0; a < s->narms; a++) {
            StudyArm *arm = &s->arms[a];
            if (i == 0) arm->tree_hash = tree_fingerprint();
            /* Wrap so we learn the exit status through a stdout-only channel:
             * run_command_timeout merges stderr and does not report it. */
            size_t need = strlen(arm->command) + 128;
            char *full = malloc(need);
            /* Subshell, not a brace group: parens need no terminating ';' after
             * a newline, and they isolate cd/env changes so run i cannot leak
             * state into run i+1. */
            snprintf(full, need, "( %s\n) ; printf '\\n" RC_SENTINEL "%%d\\n' \"$?\"",
                     arm->command);

            double t0 = time_now();
            int timed_out = 0;
            char *outp = run_command_timeout(full, STUDY_MAX_OUTPUT, s->timeout_s, &timed_out);
            double dt = time_now() - t0;
            free(full);

            StudyRun *r = &arm->runs[arm->nruns++];
            memset(r, 0, sizeof(*r));
            double run_spread = 1.0, run_lo = 0, run_hi = 0;
            r->seconds = dt;
            r->exit_code = -1;

            if (timed_out) {
                r->status = RUN_TIMEOUT;
            } else if (!outp) {
                r->status = RUN_FAILED;
            } else {
                char *sent = NULL, *scan = outp;
                while ((scan = strstr(scan, RC_SENTINEL)) != NULL) {
                    sent = scan;
                    scan += sizeof(RC_SENTINEL) - 1;
                }
                if (sent) {
                    r->exit_code = (int)strtol(sent + sizeof(RC_SENTINEL) - 1, NULL, 10);
                    *sent = '\0';   /* keep the sentinel out of the metric search */
                }
                if (r->exit_code != 0) {
                    r->status = RUN_FAILED;
                } else if (extract_metric(&re, outp, &r->value, &run_spread, &run_lo, &run_hi)) {
                    r->status = RUN_OK;
                } else {
                    r->status = RUN_NO_METRIC;
                }
                /* Keep what it printed when there is no number to report, so the
                 * next round can see WHY instead of guessing at the CLI. */
                if (r->status != RUN_OK) r->sample = squash_output(outp);
                /* Remember the worst ambiguity seen; one contaminated run is
                 * enough to invalidate the arm's numbers. */
                if (r->status == RUN_OK && run_spread > arm->metric_spread) {
                    arm->metric_spread = run_spread;
                    arm->metric_lo = run_lo;
                    arm->metric_hi = run_hi;
                }
            }
            free(outp);
            if (progress) progress(arm->name, i + 1, s->runs, ud);
        }
    }
    regfree(&re);
}

/* ── Decide ────────────────────────────────────────────────────────────── */

static const StudyArm *find_arm(const Study *s, const char *name) {
    for (int i = 0; i < s->narms; i++)
        if (strcmp(s->arms[i].name, name) == 0) return &s->arms[i];
    return NULL;
}

static double agg_value(const StudyStats *st, const char *agg) {
    if (strcmp(agg, "mean")   == 0) return st->mean;
    if (strcmp(agg, "median") == 0) return st->median;
    if (strcmp(agg, "min")    == 0) return st->min;
    if (strcmp(agg, "max")    == 0) return st->max;
    return st->sd;
}

StudyVerdict study_decide(const Study *s, char **detail) {
    if (detail) *detail = NULL;
    Rule r;
    char rerr[192];
    if (!rule_parse(s->decision_rule, &r, rerr, sizeof(rerr))) {
        if (detail) {
            char *m = malloc(320);
            snprintf(m, 320, "decision_rule could not be parsed at decide time: %s", rerr);
            *detail = m;
        }
        return VERDICT_INCONCLUSIVE;
    }

    const ROperand *ops[2] = { &r.lhs, &r.rhs };
    double vals[2];
    const StudyArm *ref[2] = { NULL, NULL };
    StudyStats stats[2];
    StringBuf d;
    sb_init(&d);

    bool insufficient = false;
    char reason[256] = {0};

    /* Arms run one after another, so an arm that rewrites the code under test
     * changes what every later arm measures — and the comparison is then between
     * two states of the world, not two configurations. Observed for real: a
     * baseline arm ran `cp work.c.bak work.c`, restoring the unoptimised file
     * before the "optimised" arm ran, so both arms measured the SAME slow code.
     * It came back REFUTED on contaminated data and a genuine 11.9x improvement
     * was reverted, with a confident and entirely wrong explanation attached. */
    /* A regex that matched several very different numbers in ONE run is not
     * identifying a metric — it is reporting whichever came last. Refuse to
     * compare arms whose numbers may not even be the same quantity. */
    for (int i = 0; i < s->narms && !insufficient; i++) {
        if (s->arms[i].metric_spread > 2.0) {
            snprintf(reason, sizeof(reason),
                "one run of arm '%s' matched both %.4g and %.4g (%.1fx apart), so the extract "
                "regex is capturing more than one quantity — anchor it to the exact field",
                s->arms[i].name, s->arms[i].metric_lo, s->arms[i].metric_hi,
                s->arms[i].metric_spread);
            insufficient = true;
        }
    }

    for (int i = 1; i < s->narms; i++) {
        if (s->arms[i].tree_hash != s->arms[0].tree_hash) {
            snprintf(reason, sizeof(reason),
                "the working tree CHANGED between arm '%s' and arm '%s' — an arm modified "
                "the files under test, so the arms did not measure the same code",
                s->arms[0].name, s->arms[i].name);
            insufficient = true;
            break;
        }
    }

    for (int i = 0; i < 2; i++) {
        if (ops[i]->is_const) { vals[i] = ops[i]->constant; continue; }
        const StudyArm *arm = find_arm(s, ops[i]->arm);
        if (!arm) {
            snprintf(reason, sizeof(reason), "arm '%s' not found", ops[i]->arm);
            insufficient = true;
            vals[i] = 0;
            continue;
        }
        ref[i] = arm;
        stats[i] = study_arm_stats(arm);
        if (stats[i].n < 2) {
            snprintf(reason, sizeof(reason),
                     "arm '%s' produced only %d valid run(s) of %d attempted",
                     arm->name, stats[i].n, arm->nruns);
            insufficient = true;
        }
        vals[i] = agg_value(&stats[i], ops[i]->agg) + ops[i]->offset;
    }

    /* Render the substituted rule regardless of outcome — showing the numbers
     * the decision was made from is the whole point. */
    char lhs_txt[128], rhs_txt[128];
    for (int i = 0; i < 2; i++) {
        char *dst = i ? rhs_txt : lhs_txt;
        if (ops[i]->is_const) snprintf(dst, 128, "%.4g", ops[i]->constant);
        else if (ops[i]->offset != 0.0)
            snprintf(dst, 128, "%s(%s)%+.4g = %.4g", ops[i]->agg, ops[i]->arm,
                     ops[i]->offset, vals[i]);
        else
            snprintf(dst, 128, "%s(%s) = %.4g", ops[i]->agg, ops[i]->arm, vals[i]);
    }

    sb_append_str(&d, "rule: ");
    sb_append_str(&d, s->decision_rule);
    sb_append_str(&d, "\nsubstituted: ");
    sb_append_str(&d, lhs_txt);
    sb_append_str(&d, " ");
    sb_append_str(&d, r.op);
    sb_append_str(&d, " ");
    sb_append_str(&d, rhs_txt);
    sb_append_char(&d, '\n');

    if (insufficient) {
        sb_append_str(&d, "verdict: INCONCLUSIVE — ");
        sb_append_str(&d, reason);
        sb_append_str(&d, ". The rule was not applied.\n");
        if (detail) *detail = sb_to_str(&d); else sb_free(&d);
        return VERDICT_INCONCLUSIVE;
    }

    bool cmp;
    if      (strcmp(r.op, "<")  == 0) cmp = vals[0] <  vals[1];
    else if (strcmp(r.op, "<=") == 0) cmp = vals[0] <= vals[1];
    else if (strcmp(r.op, ">")  == 0) cmp = vals[0] >  vals[1];
    else                              cmp = vals[0] >= vals[1];

    char line[192];
    snprintf(line, sizeof(line), "comparison: %s\n", cmp ? "holds" : "does not hold");
    sb_append_str(&d, line);

    StudyVerdict v;
    if (!cmp) {
        v = VERDICT_REFUTED;
        sb_append_str(&d, "verdict: REFUTED — the pre-registered comparison did not hold.\n");
    } else if (!r.has_p) {
        v = VERDICT_SUPPORTED;
        sb_append_str(&d, "verdict: SUPPORTED — comparison holds (no significance clause "
                          "was pre-registered, so this is a point estimate only).\n");
    } else {
        double av[STUDY_MAX_RUNS], bv[STUDY_MAX_RUNS];
        int na = arm_values(ref[0], av, STUDY_MAX_RUNS);
        int nb = arm_values(ref[1], bv, STUDY_MAX_RUNS);
        double p = study_mannwhitney_p(av, na, bv, nb);
        if (p < 0) {
            snprintf(line, sizeof(line),
                     "p: not computable (n=%d vs n=%d)\nverdict: INCONCLUSIVE\n", na, nb);
            sb_append_str(&d, line);
            v = VERDICT_INCONCLUSIVE;
        } else {
            snprintf(line, sizeof(line), "p: %.6g (Mann-Whitney U, two-sided) vs threshold %.4g\n",
                     p, r.p_threshold);
            sb_append_str(&d, line);
            if (p < r.p_threshold) {
                v = VERDICT_SUPPORTED;
                sb_append_str(&d, "verdict: SUPPORTED — comparison holds and the difference "
                                  "is distinguishable from noise.\n");
            } else {
                v = VERDICT_INCONCLUSIVE;
                sb_append_str(&d, "verdict: INCONCLUSIVE — the comparison points the right way "
                                  "but is NOT distinguishable from noise at the pre-registered "
                                  "threshold. Do not report this as a win; raise runs or find a "
                                  "bigger effect.\n");
            }
        }
    }
    if (detail) *detail = sb_to_str(&d); else sb_free(&d);
    return v;
}

/* ── Results rendering + write-back ────────────────────────────────────── */

static const char *run_status_name(StudyRunStatus s) {
    switch (s) {
        case RUN_OK:        return "ok";
        case RUN_FAILED:    return "failed";
        case RUN_TIMEOUT:   return "timeout";
        default:            return "no-metric";
    }
}

static char *render_results(const Study *s) {
    StringBuf b;
    sb_init(&b);
    char line[512];

    snprintf(line, sizeof(line), "\nmetric: %s\n\n", s->metric);
    sb_append_str(&b, line);

    sb_append_str(&b, "| arm | n ok | mean | median | sd | min | max | failed | timeout | no-metric |\n");
    sb_append_str(&b, "|-----|------|------|--------|----|-----|-----|--------|---------|-----------|\n");
    for (int i = 0; i < s->narms; i++) {
        const StudyArm *a = &s->arms[i];
        StudyStats st = study_arm_stats(a);
        int nf = 0, nt = 0, nm = 0;
        for (int j = 0; j < a->nruns; j++) {
            if (a->runs[j].status == RUN_FAILED)     nf++;
            else if (a->runs[j].status == RUN_TIMEOUT) nt++;
            else if (a->runs[j].status == RUN_NO_METRIC) nm++;
        }
        snprintf(line, sizeof(line),
            "| %s | %d/%d | %.4g | %.4g | %.4g | %.4g | %.4g | %d | %d | %d |\n",
            a->name, st.n, a->nruns, st.mean, st.median, st.sd, st.min, st.max, nf, nt, nm);
        sb_append_str(&b, line);
    }

    sb_append_str(&b, "\nRaw runs (every execution, including the ones that did not produce a number):\n\n");
    for (int i = 0; i < s->narms; i++) {
        const StudyArm *a = &s->arms[i];
        snprintf(line, sizeof(line), "- %s:", a->name);
        sb_append_str(&b, line);
        for (int j = 0; j < a->nruns; j++) {
            const StudyRun *r = &a->runs[j];
            if (r->status == RUN_OK)
                snprintf(line, sizeof(line), " %.6g", r->value);
            else
                snprintf(line, sizeof(line), " [%s rc=%d]", run_status_name(r->status), r->exit_code);
            sb_append_str(&b, line);
        }
        sb_append_char(&b, '\n');
    }

    /* For every arm that produced no usable number, show what the command
     * actually printed. A verdict of "0 valid runs" is not actionable on its
     * own — measured: three unattended rounds guessing at a CLI whose error
     * message was sitting in the discarded output. */
    bool any_diag = false;
    for (int i = 0; i < s->narms; i++) {
        const StudyArm *a = &s->arms[i];
        for (int j = 0; j < a->nruns; j++) {
            if (a->runs[j].status == RUN_OK || !a->runs[j].sample) continue;
            if (!any_diag) {
                sb_append_str(&b, "\nWhat the failing commands printed "
                                  "(first failure per arm — fix the command or the extract regex):\n\n");
                any_diag = true;
            }
            snprintf(line, sizeof(line), "- %s [%s]: %s\n",
                     a->name, run_status_name(a->runs[j].status), a->runs[j].sample);
            sb_append_str(&b, line);
            break;   /* one per arm is enough; they repeat */
        }
    }
    sb_append_char(&b, '\n');
    return sb_to_str(&b);
}

/* Replace the body under "## <name>" with `new_body`. Returns malloc'd new
 * content, or NULL if the section is absent. */
static char *replace_section(const char *content, size_t len,
                             const char *name, const char *new_body) {
    KbFrontmatter fm;
    if (kb_parse_frontmatter(content, len, &fm) < 0) return NULL;
    size_t off = fm.body_offset;
    kb_fm_free(&fm);

    size_t sec_len = 0;
    const char *sec = study_section(content + off, len - off, name, &sec_len);
    if (!sec) return NULL;

    StringBuf b;
    sb_init(&b);
    sb_append(&b, content, (size_t)(sec - content));
    sb_append_str(&b, new_body);
    const char *rest = sec + sec_len;
    sb_append(&b, rest, (size_t)(content + len - rest));
    return sb_to_str(&b);
}

/* Rewrite (or insert) a single `key:` line inside the frontmatter block only,
 * so a body line that happens to start with "status:" is never touched. */
static char *replace_fm_key(const char *content, size_t len,
                            const char *key, const char *value) {
    const char *p = content;
    const char *end = content + len;
    if (len < 4 || strncmp(p, "---", 3) != 0) return NULL;
    p += 3;
    while (p < end && *p != '\n') p++;
    if (p < end) p++;

    size_t klen = strlen(key);
    StringBuf b;
    sb_init(&b);
    sb_append(&b, content, (size_t)(p - content));
    bool wrote = false;
    while (p < end) {
        const char *le = p;
        while (le < end && *le != '\n') le++;
        size_t llen = (size_t)(le - p);
        if (llen >= 3 && strncmp(p, "---", 3) == 0) {
            if (!wrote) {                      /* key absent: insert before the closing --- */
                sb_append_str(&b, key);
                sb_append_str(&b, ": ");
                sb_append_str(&b, value);
                sb_append_char(&b, '\n');
            }
            sb_append(&b, p, (size_t)(end - p));
            break;
        }
        if (!wrote && llen > klen && strncmp(p, key, klen) == 0 && p[klen] == ':') {
            sb_append_str(&b, key);
            sb_append_str(&b, ": ");
            sb_append_str(&b, value);
            sb_append_char(&b, '\n');
            wrote = true;
        } else {
            sb_append(&b, p, llen);
            sb_append_char(&b, '\n');
        }
        p = (le < end) ? le + 1 : end;
    }
    return sb_to_str(&b);
}

static char *replace_status(const char *content, size_t len, const char *new_status) {
    return replace_fm_key(content, len, "status", new_status);
}

static char *study_path(const char *slug) {
    char *p = malloc(1024);
    if ((size_t)snprintf(p, 1024, "%s/%s.md", KB_STUDIES_DIR, slug) >= 1024) { free(p); return NULL; }
    return p;
}

char *study_run_slug(const char *slug, StudyProgressFn progress, void *ud) {
    if (!slug || !*slug) return strdup("study run: missing slug\n");
    if (!kb_slug_valid(slug)) {
        char *m = malloc(256);
        snprintf(m, 256, "study run: '%s' is not a valid slug\n", slug);
        return m;
    }
    char *path = study_path(slug);
    if (!path) return strdup("study run: path too long\n");

    size_t len = 0;
    char *content = kb_read_file(path, &len);
    if (!content) {
        char *m = malloc(512);
        snprintf(m, 512, "study run: cannot read %s (%s)\n", path, strerror(errno));
        free(path);
        return m;
    }

    Study s;
    char *perr = study_parse(content, len, &s);
    if (perr) {
        StringBuf b;
        sb_init(&b);
        sb_append_str(&b, "study run: artifact is invalid — fix these first:\n");
        sb_append_str(&b, perr);
        free(perr); free(content); free(path);
        return sb_to_str(&b);
    }

    /* Mark running so a crash mid-experiment is visible in the artifact. */
    char *marked = replace_status(content, len, "running");
    if (marked) {
        FILE *f = fopen(path, "w");
        if (f) { fputs(marked, f); fclose(f); }
        free(marked);
    }

    study_execute(&s, progress, ud);

    char *detail = NULL;
    StudyVerdict v = study_decide(&s, &detail);

    char *results = render_results(&s);
    StringBuf vb;
    sb_init(&vb);
    sb_append_char(&vb, '\n');
    sb_append_str(&vb, "**");
    sb_append_str(&vb, study_verdict_name(v));
    sb_append_str(&vb, "**\n\n```\n");
    sb_append_str(&vb, detail ? detail : "(no detail)\n");
    sb_append_str(&vb, "```\n\n");
    char *verdict_body = sb_to_str(&vb);

    /* Re-read: replace_status rewrote the file underneath us. */
    free(content);
    content = kb_read_file(path, &len);

    char *c1 = content ? replace_section(content, len, "Results", results) : NULL;
    char *c2 = c1 ? replace_section(c1, strlen(c1), "Verdict", verdict_body) : NULL;
    char *c3 = c2 ? replace_status(c2, strlen(c2), "complete") : NULL;

    const char *final = c3 ? c3 : (c2 ? c2 : (c1 ? c1 : content));
    if (final) {
        FILE *f = fopen(path, "w");
        if (f) { fputs(final, f); fclose(f); }
    }

    StringBuf out;
    sb_init(&out);
    char line[512];
    snprintf(line, sizeof(line), "Study '%s' complete: %s\n\n", slug, study_verdict_name(v));
    sb_append_str(&out, line);
    sb_append_str(&out, results);
    sb_append_str(&out, detail ? detail : "");
    snprintf(line, sizeof(line), "\nArtifact updated: %s\n", path);
    sb_append_str(&out, line);

    free(results); free(verdict_body); free(detail);
    free(c1); free(c2); free(c3); free(content); free(path);
    study_free(&s);
    return sb_to_str(&out);
}

/* ── study_write tool ──────────────────────────────────────────────────── */

char *execute_study_write(const char *body) {
    while (*body == ' ' || *body == '\t' || *body == '\n') body++;
    if (!*body)
        return strdup("study_write not allowed: missing study body. See the system prompt for the template.");

    size_t body_len = strlen(body);
    char *errs = validate_study_artifact(body, body_len);
    if (errs) {
        StringBuf out;
        sb_init(&out);
        sb_append_str(&out, "study_write rejected — fix these and resubmit:\n");
        sb_append_str(&out, errs);
        free(errs);
        return sb_to_str(&out);
    }

    KbFrontmatter fm;
    kb_parse_frontmatter(body, body_len, &fm);
    const char *slug_ref = kb_fm_get(&fm, "slug");
    if (!slug_ref) { kb_fm_free(&fm); return strdup("study_write internal: validator passed but slug missing"); }
    /* Own the slug: it points into fm, which is freed before the success
     * message is formatted. */
    char *slug = strdup(slug_ref);
    kb_fm_free(&fm);

    if (mkdir_p(KB_STUDIES_DIR) != 0) {
        char *m = malloc(256);
        snprintf(m, 256, "study_write failed: cannot create %s (%s)\n", KB_STUDIES_DIR, strerror(errno));
        free(slug);
        return m;
    }

    char *dest = study_path(slug);
    if (!dest) { free(slug); return strdup("study_write failed: destination path too long"); }

    FILE *f = fopen(dest, "w");
    if (!f) {
        char *m = malloc(512);
        snprintf(m, 512, "study_write failed: cannot write %s (%s)\n", dest, strerror(errno));
        free(slug); free(dest);
        return m;
    }
    fwrite(body, 1, body_len, f);
    if (body_len == 0 || body[body_len - 1] != '\n') fputc('\n', f);
    fclose(f);

    char *msg = malloc(640);
    snprintf(msg, 640,
        "Study written to %s. Validation passed (frontmatter, sections, arms, "
        "extract regex, decision rule).\nRun it with: basi study run %s\n"
        "The verdict will be computed from the command output, not from your reading of it.\n",
        dest, slug);
    free(dest); free(slug);
    return msg;
}

/* ── The outer loop ────────────────────────────────────────────────────── */

#define LOOP_SYSTEM_PROMPT \
"You are the hypothesis-generating half of an autonomous discovery loop.\n" \
"\n" \
"You do NOT decide what is true. A previous study was executed and its verdict was\n" \
"COMPUTED from the measured numbers by a pre-registered decision rule. Your job is\n" \
"to read that result and propose the single most informative NEXT experiment.\n" \
"\n" \
"Reply with EXACTLY ONE of:\n" \
"  (a) a complete study artifact, and nothing else; or\n" \
"  (b) the single word STOP, when the question is answered or no runnable\n" \
"      experiment would be informative.\n" \
"\n" \
"The artifact format (a YAML frontmatter block, then the sections):\n" \
"---\n" \
"slug: <ignored, it is assigned for you>\n" \
"status: proposed\n" \
"created: <date>\n" \
"metric: <what the number means>\n" \
"extract: <POSIX extended regex; capture group 1 must capture the number>\n" \
"runs: 5\n" \
"decision_rule: mean(B) < mean(A) and p < 0.05\n" \
"---\n" \
"\n" \
"## Question\n" \
"## Hypothesis\n" \
"## Experiment\n" \
"One fenced block per arm, the info string being `arm <name>`:\n" \
"```arm A\n" \
"<shell command that prints the metric>\n" \
"```\n" \
"## Results\n" \
"(pending)\n" \
"\n" \
"## Verdict\n" \
"(pending)\n" \
"\n" \
"Rules that will be enforced mechanically — violating them wastes a round:\n" \
"- Every arm must PRINT the metric, and `extract` must match what it prints.\n" \
"- runs must be at least 5. Two identical configurations on this machine differ\n" \
"  by 72-96% on some metrics, so smaller samples cannot distinguish anything.\n" \
"- The decision rule is pre-registered: you commit to it BEFORE seeing results.\n" \
"  Do not propose a rule chosen to make a favoured hypothesis come out ahead.\n" \
"- Change ONE thing between arms. An arm that differs in two ways teaches nothing.\n" \
"- A REFUTED or INCONCLUSIVE verdict is a real result. Do not retry the same\n" \
"  comparison hoping for a better roll; either test a different mechanism or STOP.\n"

/* Reject shell chaining in loop-authored arms: the allowlist below checks a
 * command's PREFIX, which means nothing if the command can chain a second one. */
static const char *arm_forbidden_chars = ";&|`$><\n";

static bool arm_command_allowed(const char *cmd, const char *allow_csv, char *why, size_t whysz) {
    for (const char *c = cmd; *c; c++) {
        if (strchr(arm_forbidden_chars, *c)) {
            snprintf(why, whysz,
                "contains '%c'; loop-authored arms may not chain, pipe, redirect or "
                "substitute commands", *c);
            return false;
        }
    }
    if (!allow_csv || !*allow_csv) {
        snprintf(why, whysz,
            "the seed study declares no allow_commands: prefixes, so no new arm "
            "command can be authorised");
        return false;
    }
    const char *p = allow_csv;
    while (*p) {
        while (*p == ' ' || *p == ',') p++;
        const char *e = p;
        while (*e && *e != ',') e++;
        size_t n = (size_t)(e - p);
        while (n && (p[n-1] == ' ' || p[n-1] == '\t')) n--;
        /* The prefix must end on a word boundary, or `./bench.sh` would also
         * authorise `./bench.sh.evil` — a different program entirely. */
        if (n && strncmp(cmd, p, n) == 0 &&
            (cmd[n] == '\0' || cmd[n] == ' ' || cmd[n] == '\t')) return true;
        p = e;
    }
    snprintf(why, whysz, "does not start with any prefix in allow_commands: %s", allow_csv);
    return false;
}

/* Pull the artifact out of the model's reply.
 *
 * Three shapes are accepted, because rejecting a well-reasoned experiment over
 * its punctuation wastes a whole round: the artifact bare, the artifact inside
 * one wrapping fence, and — the deviation Qwen3.5-9B actually produces — the
 * frontmatter emitted as a ```yaml block with the sections following it in
 * plain markdown, which is reassembled into real --- delimiters here. */
static char *extract_artifact(const char *text) {
    const char *s = text;
    while (*s == ' ' || *s == '\n' || *s == '\r' || *s == '\t') s++;

    if (strncmp(s, "---", 3) == 0) return trim_dup(s, strlen(s));

    if (strncmp(s, "```", 3) == 0) {
        const char *info = s + 3;
        const char *iend = info;
        while (*iend && *iend != '\n') iend++;
        bool is_yaml = st_icaseq(info, (size_t)(iend - info), "yaml") ||
                       st_icaseq(info, (size_t)(iend - info), "yml");
        const char *body = (*iend == '\n') ? iend + 1 : iend;
        const char *close = strstr(body, "```");
        if (close) {
            size_t n = (size_t)(close - body);
            if (is_yaml) {
                /* Frontmatter-only fence: rebuild ---, then append the rest. */
                const char *rest = close + 3;
                StringBuf b;
                sb_init(&b);
                sb_append_str(&b, "---\n");
                char *fmpart = trim_dup(body, n);
                sb_append_str(&b, fmpart);
                free(fmpart);
                sb_append_str(&b, "\n---\n");
                sb_append_str(&b, rest);
                return sb_to_str(&b);
            }
        }
        if (!is_yaml) {
            /* A wrapping ```markdown fence CONTAINS the ```arm fences, so the
             * first ``` closes an arm, not the wrapper. The real close is the
             * fence with nothing but whitespace after it; without that check the
             * artifact is truncated at ## Experiment and silently loses its arms. */
            const char *wclose = NULL;
            for (const char *q = body; (q = strstr(q, "```")) != NULL; q += 3) {
                const char *after = q + 3;
                while (*after == ' ' || *after == '\t' || *after == '\r' || *after == '\n') after++;
                if (!*after) { wclose = q; break; }
            }
            size_t n = wclose ? (size_t)(wclose - body) : strlen(body);
            char *out = trim_dup(body, n);
            if (strncmp(out, "---", 3) == 0) return out;
            free(out);
        }
    }

    /* Last resort: the artifact may start partway in, after a preamble. */
    const char *p = s;
    while (*p) {
        if (strncmp(p, "---", 3) == 0 && (p == s || p[-1] == '\n'))
            return trim_dup(p, strlen(p));
        while (*p && *p != '\n') p++;
        if (*p) p++;
    }
    return NULL;
}

/* How many times a round may re-ask after a rejected artifact. */
#define LOOP_MAX_ATTEMPTS 3

/* Generation budget for one artifact proposal. An artifact is only ~600 tokens,
 * but on a reasoning model the <think> span runs first and can consume many
 * thousands before a single line of the answer appears. This was an unexamined
 * 4096 and a seed round died with finish=length: 4096 tokens of reasoning, no
 * artifact. Budget for thinking AND writing rather than suppressing reasoning,
 * which is where hypothesis quality comes from. */
#define LOOP_MAX_TOKENS 16384

/* Incremental chat-messages JSON array. */
typedef struct { StringBuf b; int n; } Msgs;

static void msgs_init(Msgs *m) {
    sb_init(&m->b);
    sb_append_char(&m->b, '[');
    m->n = 0;
}

static void msgs_add(Msgs *m, const char *role, const char *content) {
    if (m->n++) sb_append_char(&m->b, ',');
    sb_append_str(&m->b, "{\"role\":\"");
    sb_append_str(&m->b, role);
    sb_append_str(&m->b, "\",\"content\":");
    json_escape_into(&m->b, content);   /* emits its own quotes — do not add more */
    sb_append_char(&m->b, '}');
}

static char *msgs_done(Msgs *m) {
    sb_append_char(&m->b, ']');
    return sb_to_str(&m->b);
}


/* ── Evidence grounding (gap ①: ROBIN's load-bearing pillar) ──────────────
 * ROBIN grounds every hypothesis in retrieved literature; strip that and its
 * hallucinated-citation rate goes 0%->45% (arXiv:2505.13400). BASI's proposal
 * step had the same hole — it invented mechanisms with no look at the actual
 * system (the "confident nonsense about L3 cache" false explanation). This is
 * the code-domain analog of ROBIN's Crow literature review: a bounded, READ-ONLY
 * investigation of the real system (source, symbols, and the web for prior art)
 * that returns a short factual brief. It gathers FACTS; it does not propose the
 * fix — proposing stays propose_artifact's job, exactly as ROBIN keeps its
 * literature review and its generation step separate.
 *
 * Gated by BASI_STUDY_GROUND so grounded vs ungrounded is one flag apart — which
 * is also the experimental control for measuring whether grounding actually
 * lowers the hallucinated-mechanism rate rather than assuming it. */
#define GROUND_MAX_ROUNDS 6
#define GROUND_MAX_TOKENS 8192
#define GROUND_OBS_CAP    4000   /* per-tool-result chars fed back into the chat */

static bool study_grounding_on(void) {
    const char *e = getenv("BASI_STUDY_GROUND");
    return e && *e && strcmp(e, "0") != 0;
}

#define GROUND_SYSTEM_PROMPT \
"You are the EVIDENCE step of a discovery loop. A hypothesis is about to be\n" \
"proposed for the question below. Do NOT propose it. Establish what is ACTUALLY\n" \
"TRUE about the system under test, so the hypothesis is grounded in reality\n" \
"instead of guessed from memory.\n" \
"\n" \
"Investigate with read-only tools, exactly ONE per turn, inside <tool></tool>.\n" \
"Write REAL arguments, never the placeholder brackets. Concrete examples:\n" \
"  <tool>list .</tool>                     list files in the current directory\n" \
"  <tool>read work.c</tool>                read a file (line-numbered)\n" \
"  <tool>read work.c 40 60</tool>          read 60 lines of work.c starting at line 40\n" \
"  <tool>grep \"for\" work.c</tool>          a file's lines matching a regex\n" \
"  <tool>symbols work.c</tool>             functions/types in a source file\n" \
"  <tool>web_search \"loop tiling cache\"</tool>   known techniques / prior art\n" \
"\n" \
"When the question is about local code, list then READ THE CODE before you\n" \
"web_search and before you theorise. Do not name a path you have not seen.\n" \
"\n" \
"A long file is returned one window at a time with a line count; page to a later\n" \
"section with e.g. read work.c 200 200. Re-fetching the SAME section teaches\n" \
"nothing and is rejected as a duplicate. Once you have seen the relevant code and\n" \
"the measure script, conclude — do not keep re-listing or re-reading.\n" \
"\n" \
"After a few tool calls, STOP investigating and emit the brief:\n" \
"  <evidence>\n" \
"  - <fact>; each cites a path:line, a measured number, or a source URL\n" \
"  </evidence>\n" \
"\n" \
"Report ONLY what the tools showed you — never invent a filename, number, or\n" \
"citation. If the tools found nothing useful, say that plainly. <=12 bullets.\n"

/* Body of <evidence>...</evidence>, trimmed, or NULL. Caller frees. */
static char *ground_extract_evidence(const char *text) {
    const char *o = strstr(text, "<evidence>");
    if (!o) return NULL;
    o += strlen("<evidence>");
    const char *c = strstr(o, "</evidence>");
    size_t n = c ? (size_t)(c - o) : strlen(o);
    return trim_dup(o, n);
}

/* djb2 hash, for detecting a grounding result identical to one already seen. */
static unsigned long ground_hash(const char *s) {
    unsigned long h = 5381; int c;
    while ((c = (unsigned char)*s++)) h = ((h << 5) + h) + (unsigned long)c;
    return h;
}

/* Read a line-numbered WINDOW of a text file. `start` 1-based (<=0 => 1); `count`
 * lines (<=0 => default). A long file returns one window plus a footer telling the
 * model exactly how to page to the next — so different sections are reachable, and
 * the head-only limitation of execute_readfile (cat|head -c) is gone. */
#define GROUND_WIN_DEFAULT 200
static char *ground_read_window(const char *path, int start, int count) {
    if (access(path, R_OK) != 0) {
        char *m = malloc(320);
        if (m) snprintf(m, 320, "Error: cannot read '%.240s' (does it exist? try list).", path);
        return m ? m : strdup("Error: cannot read file.");
    }
    size_t len = 0;
    char *buf = kb_read_file(path, &len);
    if (!buf) return strdup("Error: could not read file.");
    int total = 0;
    for (size_t i = 0; i < len; i++) if (buf[i] == '\n') total++;
    if (len > 0 && buf[len - 1] != '\n') total++;
    if (total == 0) total = 1;
    if (start < 1) start = 1;
    if (count < 1) count = GROUND_WIN_DEFAULT;
    int end = start + count - 1;

    StringBuf o; sb_init(&o);
    char hdr[360];
    snprintf(hdr, sizeof(hdr), "%.256s (lines %d-%d of %d):\n",
             path, start, end < total ? end : total, total);
    sb_append_str(&o, hdr);

    int line = 1;
    const char *p = buf, *fend = buf + len;
    while (p < fend && line <= end) {
        const char *nl = memchr(p, '\n', (size_t)(fend - p));
        size_t ll = nl ? (size_t)(nl - p) : (size_t)(fend - p);
        if (line >= start) {
            char num[16];
            snprintf(num, sizeof(num), "%6d\t", line);
            sb_append_str(&o, num);
            char *tmp = malloc(ll + 1);
            if (tmp) { memcpy(tmp, p, ll); tmp[ll] = '\0'; sb_append_str(&o, tmp); free(tmp); }
            sb_append_char(&o, '\n');
        }
        p = nl ? nl + 1 : fend;
        line++;
    }
    free(buf);
    if (end < total) {
        char foot[200];
        snprintf(foot, sizeof(foot),
                 "[...%d more lines below. To continue: read %.120s %d %d]\n",
                 total - end, path, end + 1, count);
        sb_append_str(&o, foot);
    }
    return sb_to_str(&o);
}

/* List a directory so the model can discover files instead of guessing paths. */
static char *ground_list_dir(const char *dir) {
    const char *d = (dir && dir[0]) ? dir : ".";
    if (strchr(d, '\'')) return strdup("Error: path must not contain a single quote.");
    char cmd[1300];
    snprintf(cmd, sizeof(cmd), "ls -la '%.1024s' 2>&1 | head -c 3000", d);
    char *r = run_command(cmd, 3400);
    return (r && r[0]) ? r : (free(r), strdup("(empty or unreadable directory)"));
}

/* Strip placeholder metacharacters a model may copy from the tool syntax
 * (<path>, [dir]) and surrounding quotes, so `read <work.c>` == `read work.c`
 * and `list [.]` == `list .`. Writes into buf, returns buf. */
static const char *ground_clean_arg(const char *a, char *buf, size_t bufsz) {
    if (!a) { buf[0] = '\0'; return buf; }
    while (*a == '<' || *a == '[' || *a == '"' || *a == '\'' || *a == ' ') a++;
    size_t n = strlen(a);
    while (n > 0 && (a[n-1] == '>' || a[n-1] == ']' || a[n-1] == '"' ||
                     a[n-1] == '\'' || a[n-1] == ' ')) n--;
    if (n >= bufsz) n = bufsz - 1;
    memcpy(buf, a, n); buf[n] = '\0';
    return buf;
}

/* Read-only tool surface for grounding. Returns a malloc'd result (caller frees). */
static char *ground_dispatch(const ArgList *al, const char **phase) {
    const char *name = al->args[0];
    char b1[1024], b2[1024], b3[64];
    const char *a1 = ground_clean_arg(al->count >= 2 ? al->args[1] : "", b1, sizeof(b1));
    const char *a2 = ground_clean_arg(al->count >= 3 ? al->args[2] : "", b2, sizeof(b2));
    const char *a3 = ground_clean_arg(al->count >= 4 ? al->args[3] : "", b3, sizeof(b3));
    if (strcasecmp(name, "list") == 0 || strcasecmp(name, "ls") == 0) {
        *phase = "list";
        return ground_list_dir(a1);
    }
    if (strcasecmp(name, "read") == 0) {
        *phase = "read";
        if (!a1[0]) return strdup("Error: read needs a path, e.g. read work.c");
        return ground_read_window(a1, a2[0] ? atoi(a2) : 0, a3[0] ? atoi(a3) : 0);
    }
    if (strcasecmp(name, "grep") == 0) {
        *phase = "grep";
        /* Keep the regex RAW — a legit pattern can start with '[' (a char class),
         * which ground_clean_arg would strip. Only the path is cleaned. */
        const char *regex = al->count >= 2 ? al->args[1] : "";
        if (!regex[0] || !a2[0])
            return strdup("Error: grep needs a regex AND a path, e.g. grep \"for\" work.c");
        return execute_readfile(a2, regex);
    }
    if (strcasecmp(name, "symbols") == 0) {
        *phase = "symbols";
        return a1[0] ? execute_symbols(a1, NULL)
                     : strdup("Error: symbols needs a path, e.g. symbols work.c");
    }
    if (strcasecmp(name, "web_search") == 0) {
        *phase = "web_search";
        return a1[0] ? execute_web_search(a1, NULL)
                     : strdup("Error: web_search needs a query: web_search \"<query>\"");
    }
    if (strcasecmp(name, "web_fetch") == 0) {
        *phase = "web_fetch";
        return a1[0] ? execute_web_fetch(a1)
                     : strdup("Error: web_fetch needs a url: web_fetch \"<url>\"");
    }
    char *m = malloc(160);
    if (m) snprintf(m, 160,
        "Error: unknown tool '%.32s'. Use list | read | grep | symbols | web_search | web_fetch.", name);
    *phase = "?";
    return m;
}

/* Bounded, read-only investigation of the system under test. Returns a malloc'd
 * evidence brief (caller frees) or NULL if grounding produced nothing usable.
 * `context` is the same text the proposer will see (the question or prior verdict). */
typedef struct { const char *role; char *content; } GTurn;
static char *study_ground(int port, const char *context, StringBuf *log) {
    /* Keep the native tool grammar out of the <tool> ReAct format, same reason
     * deepsearch does: a native-tools model otherwise emits its trained tool-call
     * syntax and extract_tool_call (legacy <tool>) can't parse it. Restored below. */
    int prev_suppress = basi_srv_suppress_grammar;
    basi_srv_suppress_grammar = 1;
    /* Grounding is mechanical navigation (read/grep/list) — a <think> block per
     * tool call is pure latency here (measured ~2 min/round on a reasoning model).
     * Scope enable_thinking=false to this loop only; the hypothesis step that
     * consumes the brief keeps full reasoning. Restored below. */
    int prev_no_think = basi_srv_no_think;
    basi_srv_no_think = 1;
    int prev_tool_n = basi_tools_registered();
    basi_set_tools(NULL, 0);

    GTurn *turns = NULL; int nt = 0, cap = 0;
#define GPUSH(R,C) do { \
        if (nt == cap) { cap = cap ? cap*2 : 8; turns = realloc(turns, (size_t)cap*sizeof(*turns)); } \
        turns[nt].role = (R); turns[nt].content = (C); nt++; } while (0)

    GPUSH("system", strdup(GROUND_SYSTEM_PROMPT));
    {
        StringBuf u; sb_init(&u);
        sb_append_str(&u, "Question the hypothesis will address:\n\n");
        sb_append_str(&u, context);
        GPUSH("user", sb_to_str(&u));
    }

    /* Hashes of results already returned, so a re-fetch of the SAME section is
     * caught and nudged rather than silently burning a round. Reading a DIFFERENT
     * section of a long file produces different bytes → different hash → allowed. */
    unsigned long *seen = NULL; int nseen = 0, capseen = 0;

    char *evidence = NULL;
    printf("\033[36m[study/ground] investigating the system before proposing "
           "(up to %d tool calls)\033[0m\n", GROUND_MAX_ROUNDS);
    fflush(stdout);

    for (int round = 1; round <= GROUND_MAX_ROUNDS && !evidence; round++) {
        /* On the final round, force the brief instead of another tool. */
        if (round == GROUND_MAX_ROUNDS && nt > 0 && strcmp(turns[nt-1].role, "user") == 0) {
            const char *force = "\n\n[Investigation budget reached — emit the "
                "<evidence>...</evidence> brief now from what you found; do not call another tool.]";
            size_t ol = strlen(turns[nt-1].content);
            char *nc = malloc(ol + strlen(force) + 1);
            if (nc) { memcpy(nc, turns[nt-1].content, ol); strcpy(nc + ol, force);
                      free(turns[nt-1].content); turns[nt-1].content = nc; }
        }

        Msgs m; msgs_init(&m);
        for (int i = 0; i < nt; i++) msgs_add(&m, turns[i].role, turns[i].content);
        char *msgs = msgs_done(&m);

        SrvChatResult *r = srvchat_complete(port, msgs, NULL, NULL, GROUND_MAX_TOKENS, NULL, NULL, NULL);
        free(msgs);
        if (!r || !r->content || !*r->content) { if (r) srvchat_free(r); break; }
        char *reply = trim_dup(r->content, strlen(r->content));
        srvchat_free(r);
        GPUSH("assistant", strdup(reply));

        evidence = ground_extract_evidence(reply);
        if (evidence) { free(reply); break; }

        size_t tlen = 0;
        const char *tbody = extract_tool_call(reply, &tlen);
        if (!tbody) {
            GPUSH("user", strdup("Emit exactly one <tool>...</tool> action, or the "
                                 "<evidence>...</evidence> brief if you have enough facts."));
            free(reply);
            continue;
        }
        char *cmd = malloc(tlen + 1);
        memcpy(cmd, tbody, tlen); cmd[tlen] = '\0';
        ArgList al = tokenize_command(cmd);
        free(cmd);
        if (al.count < 1) {
            arglist_free(&al);
            GPUSH("user", strdup("That tool call was empty. Use e.g. <tool>read src/foo.c</tool>."));
            free(reply);
            continue;
        }
        const char *phase = "?";
        char *raw = ground_dispatch(&al, &phase);
        printf("\033[90m[study/ground %d/%d] %s: %.70s\033[0m\n",
               round, GROUND_MAX_ROUNDS, phase, al.count >= 2 ? al.args[1] : "");
        fflush(stdout);
        arglist_free(&al);
        if (!raw) raw = strdup("(no result)");

        /* Same-section re-read guard: identical bytes to a prior result → no new
         * information. Nudge toward a different section / file / the brief instead
         * of feeding the duplicate back (which is what made it churn). */
        unsigned long h = ground_hash(raw);
        bool dup = false;
        for (int i = 0; i < nseen; i++) if (seen[i] == h) { dup = true; break; }
        if (dup) {
            printf("\033[33m[study/ground %d/%d] duplicate result — nudging\033[0m\n",
                   round, GROUND_MAX_ROUNDS);
            fflush(stdout);
            GPUSH("user", strdup(
                "That returned content IDENTICAL to a result you already have — no new "
                "information. Do not repeat it. Read a DIFFERENT section (read <path> "
                "<start> <count>), investigate a different file, or emit the "
                "<evidence>...</evidence> brief now."));
            free(raw); free(reply);
            continue;
        }
        if (nseen == capseen) { capseen = capseen ? capseen * 2 : 8;
                                seen = realloc(seen, (size_t)capseen * sizeof(*seen)); }
        seen[nseen++] = h;

        StringBuf ob; sb_init(&ob);
        sb_append_str(&ob, "Result:\n");
        if (strlen(raw) > GROUND_OBS_CAP) {
            char *tmp = malloc(GROUND_OBS_CAP + 1);
            if (tmp) { memcpy(tmp, raw, GROUND_OBS_CAP); tmp[GROUND_OBS_CAP] = '\0';
                       sb_append_str(&ob, tmp); free(tmp); }
            sb_append_str(&ob, "\n...[truncated]");
        } else {
            sb_append_str(&ob, raw);
        }
        free(raw);
        GPUSH("user", sb_to_str(&ob));
        free(reply);
    }

    if (evidence && log) {
        sb_append_str(log, "\n-- grounded evidence --\n");
        sb_append_str(log, evidence);
        sb_append_char(log, '\n');
    }
    if (!evidence) {
        sb_append_str(log, "\n[study/ground] no evidence brief produced; proposing ungrounded.\n");
        printf("\033[33m[study/ground] no brief produced; proposing ungrounded\033[0m\n");
        fflush(stdout);
    }
    for (int i = 0; i < nt; i++) free(turns[i].content);
    free(turns);
    free(seen);
#undef GPUSH

    basi_srv_suppress_grammar = prev_suppress;
    basi_srv_no_think = prev_no_think;
    if (prev_tool_n > 0) { int n; const BasiToolDef *d = basi_tool_defs(&n); basi_set_tools(d, n); }

    if (evidence && !*evidence) { free(evidence); evidence = NULL; }
    return evidence;
}

/* Ask the model for a study artifact and keep asking, up to LOOP_MAX_ATTEMPTS,
 * feeding the validator's own errors back each time. Returns a malloc'd,
 * VALIDATED and safety-gated artifact, or NULL. *said_stop is set when the model
 * deliberately ended the loop rather than failing.
 *
 * Shared by the seed step (from a question) and every later round (from the
 * previous verdict), so both get the same self-correction. Measured need: three
 * consecutive single-shot attempts on Qwen3.5-9B produced three DIFFERENT
 * malformed artifacts, and catching them without a retry only ends the run. */
static char *propose_artifact(const StudyLoopOpts *opts, const char *user_msg,
                              const char *target_slug, const char *parent_slug,
                              const char *allow, StringBuf *log, bool *said_stop) {
    char *prev_reply = NULL, *feedback = NULL, *out = NULL;
    char line[1024];
    if (said_stop) *said_stop = false;

    /* Gap ①: ground the hypothesis in the real system before proposing it. Done
     * ONCE (not per retry attempt) — the evidence does not change between retries. */
    char *evidence = study_grounding_on() ? study_ground(opts->port, user_msg, log) : NULL;

    for (int attempt = 1; attempt <= LOOP_MAX_ATTEMPTS && !out; attempt++) {
        Msgs m;
        msgs_init(&m);
        msgs_add(&m, "system", LOOP_SYSTEM_PROMPT);

        /* Prior rounds, pulled from BASI's retrieval memory rather than carried
         * in the prompt. Without this the loop is amnesiac: each round sees only
         * the round before it, so it re-proposes approaches that already failed
         * — measured, four rounds against llama-bench failing the same way with
         * no record that the previous three had. Retrieval keeps the prompt
         * BOUNDED while history grows, which is what lets the loop run
         * indefinitely instead of until the context fills. */
        if (mem_count() > 0) {
            char *hits[6]; float sc[6];
            int nh = mem_retrieve(user_msg, 6, 0.25f, hits, sc);
            if (nh > 0) {
                StringBuf hb;
                sb_init(&hb);
                sb_append_str(&hb, "Earlier rounds of this investigation (most relevant first). "
                                   "Do NOT repeat an approach that already failed here — either "
                                   "fix what broke it or test something else:\n\n");
                for (int h = 0; h < nh; h++) {
                    sb_append_str(&hb, hits[h]);
                    sb_append_char(&hb, '\n');
                    free(hits[h]);
                }
                char *hs = sb_to_str(&hb);
                msgs_add(&m, "user", hs);
                free(hs);
            }
        }

        /* Inject the grounded evidence brief just before the proposal request, so
         * the model writes its hypothesis, metric, and arms FROM these facts —
         * ROBIN interpolates its literature review into the generation prompt the
         * same way. */
        if (evidence && *evidence) {
            StringBuf eb; sb_init(&eb);
            sb_append_str(&eb, "GROUNDED EVIDENCE about the system under test, gathered by "
                               "read-only investigation just now. Base the hypothesis, the "
                               "metric, and the arms on these FACTS — not on assumptions:\n\n");
            sb_append_str(&eb, evidence);
            char *es = sb_to_str(&eb);
            msgs_add(&m, "user", es);
            free(es);
        }

        msgs_add(&m, "user", user_msg);
        if (prev_reply && feedback) {
            msgs_add(&m, "assistant", prev_reply);
            StringBuf fb;
            sb_init(&fb);
            sb_append_str(&fb, "That artifact was REJECTED:\n\n");
            sb_append_str(&fb, feedback);
            sb_append_str(&fb, "\nResend the COMPLETE corrected artifact, starting with the "
                               "--- frontmatter line. Do not explain, do not apologise, do "
                               "not send only the changed part.\n");
            char *fbs = sb_to_str(&fb);
            msgs_add(&m, "user", fbs);
            free(fbs);
        }
        char *msgs = msgs_done(&m);

        SrvChatResult *r = srvchat_complete(opts->port, msgs, NULL, NULL, LOOP_MAX_TOKENS, NULL, NULL, NULL);
        free(msgs);
        if (!r || !r->content || !*r->content) {
            if (r && r->reasoning && *r->reasoning)
                snprintf(line, sizeof(line),
                    "loop: model produced %d tokens of reasoning but no answer (finish=%s); "
                    "stopping. Try BASI_NO_THINK=1.\n",
                    r->completion_tokens, r->finish_reason ? r->finish_reason : "?");
            else
                snprintf(line, sizeof(line),
                    "loop: no response from llama-server on port %d; stopping.\n", opts->port);
            sb_append_str(log, line);
            if (r) srvchat_free(r);
            break;
        }

        char *artifact = extract_artifact(r->content);
        char *reply = trim_dup(r->content, strlen(r->content));
        srvchat_free(r);

        if (!artifact && strncmp(reply, "STOP", 4) == 0) {
            if (said_stop) *said_stop = true;
            free(reply);
            break;
        }

        char *a1 = NULL, *a2 = NULL, *a3 = NULL, *a4 = NULL, *c = NULL, *why_bad = NULL;
        if (!artifact) {
            why_bad = strdup("- the reply did not contain a study artifact. It must start "
                             "with a --- frontmatter line.\n");
        } else {
            a1 = replace_fm_key(artifact, strlen(artifact), "slug", target_slug);
            a2 = a1 ? replace_fm_key(a1, strlen(a1), "status", "proposed") : NULL;
            a3 = (a2 && parent_slug) ? replace_fm_key(a2, strlen(a2), "parent", parent_slug) : NULL;
            char *base_for_allow = a3 ? a3 : a2;
            a4 = (base_for_allow && allow)
               ? replace_fm_key(base_for_allow, strlen(base_for_allow), "allow_commands", allow) : NULL;
            c = a4 ? a4 : (a3 ? a3 : (a2 ? a2 : a1));
            if (!c) c = artifact;

            why_bad = validate_study_artifact(c, strlen(c));

            if (!why_bad && !opts->unsafe) {
                StudyArm arms[STUDY_MAX_ARMS];
                char *aerr = NULL;
                KbFrontmatter cfm;
                kb_parse_frontmatter(c, strlen(c), &cfm);
                int n = study_parse_arms(c + cfm.body_offset, strlen(c) - cfm.body_offset,
                                         arms, STUDY_MAX_ARMS, &aerr);
                kb_fm_free(&cfm);
                free(aerr);
                StringBuf gb;
                sb_init(&gb);
                for (int i = 0; i < n; i++) {
                    char why[256];
                    if (!arm_command_allowed(arms[i].command, allow, why, sizeof(why))) {
                        snprintf(line, sizeof(line),
                            "- arm '%s' is not authorised: %s\n  command: %s\n",
                            arms[i].name, why, arms[i].command);
                        sb_append_str(&gb, line);
                        fputs(line, stderr);
                    }
                    free(arms[i].name); free(arms[i].command);
                }
                if (gb.len) why_bad = sb_to_str(&gb); else sb_free(&gb);
            }
        }

        if (!why_bad) {
            out = strdup(c);
            free(reply);
        } else {
            snprintf(line, sizeof(line), "\nloop: attempt %d/%d rejected:\n",
                     attempt, LOOP_MAX_ATTEMPTS);
            sb_append_str(log, line);
            sb_append_str(log, why_bad);
            fputs(line, stderr);
            free(prev_reply); prev_reply = reply;
            free(feedback);   feedback = why_bad;
        }
        free(artifact); free(a1); free(a2); free(a3); free(a4);
    }
    free(prev_reply); free(feedback); free(evidence);
    return out;
}

/* ── Ranking across trajectories ────────────────────────────────────────
 * ROBIN closes with Bradley-Terry-Luce over pairwise LLM preferences, because
 * it cannot run its experiments — model opinion is the only signal it has. Here
 * every finding already carries measured numbers and a p-value, so the ranking
 * is arithmetic over those instead: effect size, with significance as the gate.
 * The ordering is not something the model can argue with. */
typedef struct {
    char   slug[192];
    bool   supported;
    bool   measured;     /* both arms produced numbers — a real negative counts */
    double effect_pct;   /* |better - worse| / worse * 100 */
    double p;
    double best, other;
    char   verdict[32];  /* SUPPORTED / REFUTED / INCONCLUSIVE */
    char   what[320];    /* what the arms actually differed by */
    char   detail[320];
} TrajFinding;

/* Pull the executed numbers back out of a completed artifact. */
static bool summarize_study(const char *slug, TrajFinding *out) {
    memset(out, 0, sizeof(*out));
    snprintf(out->slug, sizeof(out->slug), "%s", slug ? slug : "?");
    char *path = study_path(slug);
    size_t len = 0;
    char *c = path ? kb_read_file(path, &len) : NULL;
    free(path);
    if (!c) return false;

    out->supported = strstr(c, "**SUPPORTED**") != NULL;

    const char *pp = strstr(c, "\np: ");
    if (pp) out->p = strtod(pp + 4, NULL);

    /* "substituted: mean(B) = 75.49 > mean(A) = 74.31" */
    const char *sub = strstr(c, "substituted: ");
    if (sub) {
        const char *e = strchr(sub, '\n');
        size_t n = e ? (size_t)(e - sub) : strlen(sub);
        if (n >= sizeof(out->detail)) n = sizeof(out->detail) - 1;
        memcpy(out->detail, sub, n);
        out->detail[n] = '\0';

        double v[4]; int nv = 0;
        for (const char *q = sub; q < sub + n && nv < 4; q++) {
            if (*q == '=' ) {
                char *end = NULL;
                double d = strtod(q + 1, &end);
                if (end != q + 1) { v[nv++] = d; q = end - 1; }
            }
        }
        if (nv >= 2) {
            double a = v[0], b = v[1];
            out->best  = a > b ? a : b;
            out->other = a > b ? b : a;
            out->measured = true;
            if (out->other != 0.0)
                out->effect_pct = (out->best - out->other) / out->other * 100.0;
        }
    }

    snprintf(out->verdict, sizeof(out->verdict), "%s",
             out->supported                      ? "SUPPORTED"
           : strstr(c, "**REFUTED**")            ? "REFUTED"
                                                 : "INCONCLUSIVE");

    /* What the arms actually differed by. A finding nobody can read is not a
     * finding: reporting only a slug and "not supported" hides the entire
     * content of a negative result. */
    KbFrontmatter fm;
    if (kb_parse_frontmatter(c, len, &fm) >= 0) {
        StudyArm arms[STUDY_MAX_ARMS];
        char *aerr = NULL;
        int n = study_parse_arms(c + fm.body_offset, len - fm.body_offset,
                                 arms, STUDY_MAX_ARMS, &aerr);
        /* Strip the common prefix so only the distinguishing tail remains. */
        size_t common = 0;
        if (n >= 2) {
            while (arms[0].command[common] && arms[1].command[common] &&
                   arms[0].command[common] == arms[1].command[common]) common++;
            while (common > 0 && arms[0].command[common - 1] != ' ') common--;
        }
        size_t used = 0;
        for (int i = 0; i < n && used + 8 < sizeof(out->what); i++) {
            const char *tailp = arms[i].command + (strlen(arms[i].command) > common ? common : 0);
            int w = snprintf(out->what + used, sizeof(out->what) - used,
                             "%s%s=[%.90s]", used ? "  " : "", arms[i].name,
                             *tailp ? tailp : "(same)");
            if (w > 0) used += (size_t)w;
        }
        for (int i = 0; i < n; i++) { free(arms[i].name); free(arms[i].command); }
        free(aerr);
        kb_fm_free(&fm);
    }
    free(c);
    return true;
}

static int cmp_finding(const void *x, const void *y) {
    const TrajFinding *a = x, *b = y;
    if (a->supported != b->supported) return a->supported ? -1 : 1;  /* supported first */
    if (a->effect_pct < b->effect_pct) return 1;                     /* bigger effect first */
    if (a->effect_pct > b->effect_pct) return -1;
    return 0;
}
/* One trajectory: seed a study (optionally from a question), then chain rounds
 * off each computed verdict until STOP or the round budget. `explored` (may be
 * NULL) lists what EARLIER trajectories already covered, so this one is pushed
 * somewhere else. Appends its own one-line finding to `finding` if non-NULL. */
static char *study_trajectory(const char *seed_slug, const StudyLoopOpts *opts,
                              const char *explored, const char *angle, StringBuf *finding,
                              char **final_slug,
                              TrajFinding *finds, int *nfind, int maxfind) {
    StringBuf log;
    sb_init(&log);
    char line[1024];

    char base[256];
    snprintf(base, sizeof(base), "%s", seed_slug);
    char *cur_slug = strdup(seed_slug);
    char *last_done_slug = NULL;

    /* The allowlist is read ONCE, from the seed a human wrote, and is injected
     * into every generated round. Re-reading it per round would let the model
     * widen its own authority simply by writing a broader allow_commands. */
    char *allow = NULL;
    {
        char *sp = study_path(seed_slug);
        size_t sl = 0;
        char *sc = sp ? kb_read_file(sp, &sl) : NULL;
        if (sc) {
            KbFrontmatter sfm;
            if (kb_parse_frontmatter(sc, sl, &sfm) >= 0) {
                const char *a = kb_fm_get(&sfm, "allow_commands");
                if (a) allow = strdup(a);
                kb_fm_free(&sfm);
            }
            free(sc);
        }
        free(sp);
    }
    /* --question: no seed artifact exists yet, so the model writes the FIRST
     * study too — question, metric, extract regex, arms, decision rule. This is
     * what makes the loop general-purpose rather than a runner for experiments a
     * human already designed. The allowlist comes from the CLI, not from the
     * artifact: a model that writes its own allowlist has no allowlist. */
    if (opts->question && *opts->question) {
        allow = opts->allow ? strdup(opts->allow) : NULL;
        if (!allow && !opts->unsafe) {
            sb_free(&log);
            free(cur_slug);
            return strdup(
                "study loop --question needs --allow: the model writes the whole first study,\n"
                "including its arm commands, so the list of permitted command prefixes must\n"
                "come from you. e.g. --allow './bench.sh, python3 tools/measure.py'\n"
                "Pass --unsafe to run without one.\n");
        }

        StringBuf u;
        sb_init(&u);
        sb_append_str(&u, "Design the FIRST study for this question:\n\n");
        sb_append_str(&u, opts->question);
        sb_append_str(&u, "\n\nYou are choosing what to measure and how. Pick a metric that the "
                          "command actually prints, an extract regex that matches it, two arms "
                          "differing in exactly ONE thing, and a decision rule you commit to "
                          "before seeing any data.\n");
        /* Per-trajectory lens: shapes what THIS investigator looks at first so
         * independent trajectories spread out instead of all chasing the model's
         * favourite knob. A bias, not a rule — convergence on real evidence is fine. */
        if (angle && *angle) {
            sb_append_str(&u, "\nYOUR INVESTIGATIVE LENS (this shapes what you look at FIRST so "
                              "independent investigators spread out rather than all testing the same "
                              "obvious thing; it is a bias, not a rule — follow the evidence, and if "
                              "it points where another investigator would also look, that convergence "
                              "is corroboration worth having): ");
            sb_append_str(&u, angle);
            sb_append_char(&u, '\n');
        }
        /* ROBIN generates N *distinct* ideas rather than iterating one — breadth
         * first, then rank. Without this a trajectory re-derives the previous
         * one's finding: the first run here found flash attention worth +1.5%
         * and stopped, never touching batch size, threads or offload. */
        if (explored && *explored) {
            sb_append_str(&u, "\nEARLIER TRAJECTORIES ALREADY INVESTIGATED THIS QUESTION AND "
                              "COVERED THE GROUND BELOW. Investigate a DIFFERENT dimension — do "
                              "not re-test these variables, even if a result there looked "
                              "promising:\n\n");
            sb_append_str(&u, explored);
            sb_append_char(&u, '\n');
        }
        if (allow && !opts->unsafe) {
            sb_append_str(&u, "\nEvery arm command MUST begin with one of these exact prefixes, "
                              "and may not chain, pipe or redirect: ");
            sb_append_str(&u, allow);
            sb_append_char(&u, '\n');
        }
        char *user_msg = sb_to_str(&u);

        snprintf(line, sizeof(line), "\n══ seeding from question ══\n");
        sb_append_str(&log, line);
        fputs(line, stderr);

        bool stop = false;
        char *seed = propose_artifact(opts, user_msg, seed_slug, NULL, allow, &log, &stop);
        free(user_msg);
        if (!seed) {
            sb_append_str(&log, "\nloop: could not design a study for that question; stopping.\n");
            free(allow); free(cur_slug);
            return sb_to_str(&log);
        }
        char *msg = execute_study_write(seed);
        sb_append_str(&log, msg);
        bool ok = strstr(msg, "rejected") == NULL && strstr(msg, "failed") == NULL;
        free(msg); free(seed);
        if (!ok) {
            sb_append_str(&log, "loop: could not persist the seed study; stopping.\n");
            free(allow); free(cur_slug);
            return sb_to_str(&log);
        }
    }

    if (!allow && !opts->unsafe) {
        sb_free(&log);
        free(cur_slug);
        return strdup(
            "study loop: the seed declares no allow_commands: prefixes.\n"
            "The loop executes model-authored shell commands unattended, so it needs an\n"
            "explicit allowlist of permitted command prefixes, e.g.\n"
            "  allow_commands: ./bench.sh, python3 tools/measure.py\n"
            "Pass --unsafe to run without one.\n");
    }

    /* max_rounds <= 0 means run until the model says STOP or something breaks.
     * Safe now that history lives in retrieval memory rather than the prompt:
     * the context per round is bounded, so nothing grows without limit. */
    for (int round = 1; opts->max_rounds <= 0 || round <= opts->max_rounds; round++) {
        snprintf(line, sizeof(line),
                 "\n══ round %d/%d — study '%s' ══\n", round, opts->max_rounds, cur_slug);
        sb_append_str(&log, line);
        fputs(line, stderr);

        /* 1. Execute. Deterministic: this is where the loop touches ground. */
        char *report = study_run_slug(cur_slug, cli_progress, NULL);
        free(last_done_slug); last_done_slug = strdup(cur_slug);
        /* Record EVERY round, not just the last. A trajectory's earlier rounds
         * carry real results — "this knob does nothing" is a finding — and
         * reporting only the final one threw them away. */
        if (finds && nfind && *nfind < maxfind &&
            summarize_study(cur_slug, &finds[*nfind])) (*nfind)++;
        sb_append_str(&log, report);

        char *path = study_path(cur_slug);
        size_t len = 0;
        char *content = path ? kb_read_file(path, &len) : NULL;

        /* Record what this round tried and what came of it, so later rounds can
         * retrieve it instead of rediscovering it. Compact on purpose: the
         * hypothesis, the exact arm commands, and the computed verdict line —
         * that is what a future round needs in order not to repeat this one. */
        if (content) {
            StringBuf rec;
            sb_init(&rec);
            snprintf(line, sizeof(line), "Round %d — study '%s'\n", round, cur_slug);
            sb_append_str(&rec, line);
            size_t hl = 0;
            const char *hyp = study_section(content, len, "Hypothesis", &hl);
            if (hyp && hl) {
                char *h = trim_dup(hyp, hl > 400 ? 400 : hl);
                sb_append_str(&rec, "hypothesis: "); sb_append_str(&rec, h);
                sb_append_char(&rec, '\n'); free(h);
            }
            StudyArm ra[STUDY_MAX_ARMS];
            char *rerr = NULL;
            KbFrontmatter rfm;
            if (kb_parse_frontmatter(content, len, &rfm) >= 0) {
                int rn = study_parse_arms(content + rfm.body_offset, len - rfm.body_offset,
                                          ra, STUDY_MAX_ARMS, &rerr);
                for (int i = 0; i < rn; i++) {
                    char *rab = abbrev_cmd(ra[i].command);
                    snprintf(line, sizeof(line), "arm %s: %s\n", ra[i].name, rab);
                    free(rab);
                    sb_append_str(&rec, line);
                    free(ra[i].name); free(ra[i].command);
                }
                free(rerr);
                kb_fm_free(&rfm);
            }
            const char *v = strstr(content, "verdict: ");
            if (v) {
                const char *ve = strchr(v, '\n');
                char *vs = trim_dup(v, ve ? (size_t)(ve - v) : strlen(v));
                sb_append_str(&rec, vs); sb_append_char(&rec, '\n');
                free(vs);
            }
            char *recs = sb_to_str(&rec);
            mem_add(recs);
            free(recs);
        }
        if (!content) {
            sb_append_str(&log, "loop: cannot re-read the artifact; stopping.\n");
            free(report); free(path); break;
        }

        if (opts->max_rounds > 0 && round == opts->max_rounds) {
            sb_append_str(&log, "\nloop: round budget exhausted.\n");
            free(report); free(content); free(path);
            break;
        }

        /* 2. Ask for the next hypothesis, given the COMPUTED verdict. */
        char next_slug[288];
        snprintf(next_slug, sizeof(next_slug), "%s-r%d", base, round + 1);

        StringBuf u;
        sb_init(&u);
        sb_append_str(&u, "The study just executed. Here is the artifact, including the "
                          "measured Results and the computed Verdict:\n\n");
        sb_append(&u, content, len);
        sb_append_str(&u, "\n\nPropose the next study artifact, or reply STOP.\n");
        if (angle && *angle) {
            sb_append_str(&u, "\nStay within your investigative lens: ");
            sb_append_str(&u, angle);
            sb_append_char(&u, '\n');
        }
        if (allow && !opts->unsafe) {
            sb_append_str(&u, "\nEvery arm command MUST begin with one of these exact "
                              "prefixes, and may not chain, pipe or redirect: ");
            sb_append_str(&u, allow);
            sb_append_char(&u, '\n');
        }
        char *user_msg = sb_to_str(&u);

        bool said_stop = false;
        char *cand = propose_artifact(opts, user_msg, next_slug, cur_slug, allow, &log, &said_stop);
        free(user_msg);
        free(report); free(content); free(path);

        if (said_stop) {
            sb_append_str(&log, "\nloop: model reported the question answered (STOP).\n");
            free(cand);
            break;
        }
        if (!cand) {
            snprintf(line, sizeof(line),
                "\nloop: no valid study after %d attempts; stopping.\n", LOOP_MAX_ATTEMPTS);
            sb_append_str(&log, line);
            break;
        }

        /* 3. Write it and continue. */
        char *msg = execute_study_write(cand);
        sb_append_str(&log, "\n");
        sb_append_str(&log, msg);
        bool wrote = strstr(msg, "rejected") == NULL && strstr(msg, "failed") == NULL;
        free(msg); free(cand);
        if (!wrote) { sb_append_str(&log, "loop: could not persist the next study; stopping.\n"); break; }

        free(cur_slug);
        cur_slug = strdup(next_slug);
    }

    /* Hand the next trajectory a compact account of what this one covered:
     * the variable it varied (its arm commands) and what the verdict was. */
    if (finding && last_done_slug) {
        char *fp = study_path(last_done_slug);
        size_t fl = 0;
        char *fc = fp ? kb_read_file(fp, &fl) : NULL;
        if (fc) {
            KbFrontmatter ffm;
            if (kb_parse_frontmatter(fc, fl, &ffm) >= 0) {
                StudyArm fa[STUDY_MAX_ARMS];
                char *ferr = NULL;
                int fn = study_parse_arms(fc + ffm.body_offset, fl - ffm.body_offset,
                                          fa, STUDY_MAX_ARMS, &ferr);
                snprintf(line, sizeof(line), "- trajectory '%s':\n", last_done_slug);
                sb_append_str(finding, line);
                for (int i = 0; i < fn; i++) {
                    char *ab = abbrev_cmd(fa[i].command);
                    snprintf(line, sizeof(line), "    arm %s: %s\n", fa[i].name, ab);
                    sb_append_str(finding, line);
                    free(ab);
                    free(fa[i].name); free(fa[i].command);
                }
                free(ferr);
                kb_fm_free(&ffm);
            }
            const char *v = strstr(fc, "verdict: ");
            if (v) {
                const char *ve = strchr(v, '\n');
                char *vs = trim_dup(v, ve ? (size_t)(ve - v) : strlen(v));
                snprintf(line, sizeof(line), "    %.400s\n", vs);
                sb_append_str(finding, line);
                free(vs);
            }
            free(fc);
        }
        free(fp);
    }
    if (final_slug) *final_slug = last_done_slug; else free(last_done_slug);
    free(cur_slug);
    free(allow);
    return sb_to_str(&log);
}


/* Shared final synthesis for both the serial and concurrent trajectory paths:
 * order findings by measured effect (supported first, largest effect first) and
 * report the measured negatives too. Nothing here is a judgement call — the
 * ordering is arithmetic over the numbers, see cmp_finding. */
static void study_rank_synthesis(StringBuf *log, TrajFinding *finds, int nfind) {
    if (nfind <= 1) return;
    qsort(finds, (size_t)nfind, sizeof(finds[0]), cmp_finding);
    sb_append_str(log, "\n════ findings, ranked by measured effect ════\n\n");
    char rl[1200];
    int shown = 0;
    for (int i = 0; i < nfind; i++) {
        if (!finds[i].supported) continue;
        snprintf(rl, sizeof(rl), "%d. WORKS  %+.1f%%  (%.4g vs %.4g, p=%.3g)\n"
                                 "     tested: %.300s\n     %.180s\n",
                 ++shown, finds[i].effect_pct, finds[i].best, finds[i].other,
                 finds[i].p, finds[i].what, finds[i].slug);
        sb_append_str(log, rl);
    }
    if (!shown)
        sb_append_str(log, "Nothing beat the baseline. The measured negatives:\n\n");

    /* Negative results are results. A knob that provably does nothing is worth
     * as much as one that works — it stops the next person retrying it — so
     * report what was tested and the numbers, not just a slug. */
    for (int i = 0; i < nfind; i++) {
        if (finds[i].supported) continue;
        if (finds[i].measured)
            snprintf(rl, sizeof(rl), "   %-12s no gain (%.4g vs %.4g)\n     tested: %.300s\n",
                     finds[i].verdict, finds[i].best, finds[i].other, finds[i].what);
        else
            snprintf(rl, sizeof(rl), "   %-12s no usable measurement\n     tested: %.300s\n",
                     finds[i].verdict, finds[i].what);
        sb_append_str(log, rl);
    }
}

/* Children in flight at once. A modest cap: each is a full study (LLM calls plus
 * shell builds), and the server's KV slots (-np N) are the real ceiling anyway.
 * More trajectories than this run in sequential waves. */
#define STUDY_MAX_CONCURRENT 6

/* Per-trajectory investigative lenses. Concurrent trajectories otherwise share one
 * identical seed prompt and HERD on whatever dimension the model favours — measured
 * ~1 duplicate per 4 trajectories, p=0.012, all piling onto the same knob. A different
 * lens per trajectory spreads the search. Crucially these BIAS, they do not FORBID:
 * two independent lenses that still converge on the same variable are corroborating
 * it — a signal worth having; mechanical herding on a shared prompt is not. Assigned
 * by trajectory index in BOTH loops. Domain-general on purpose — the loop investigates
 * any question in any field. */
static const char *const STUDY_ANGLES[] = {
    "Go after the single change you believe has the LARGEST effect on the metric, even if it is the obvious one.",
    "Investigate a variable the obvious approaches would SKIP — the overlooked, unglamorous, or non-obvious lever.",
    "Test an INTERACTION or second-order effect — two things together, or a knock-on consequence — not one knob alone.",
    "Question the BASELINE or a stated ASSUMPTION: is it measured correctly, and does a 'known' fact actually hold here?",
    "Find the CHEAPEST viable win — the lowest-effort, lowest-risk change that could still move the metric.",
    "Probe a LIMIT or failure mode: push a variable to its extreme, or test something you expect NOT to help, to rule it out.",
};
enum { STUDY_NUM_ANGLES = (int)(sizeof(STUDY_ANGLES) / sizeof(STUDY_ANGLES[0])) };

/* ── Per-child working-tree isolation ──────────────────────────────────
 * Concurrent arms run shell commands in the working directory, so two
 * trajectories that write the same relative path would corrupt each other's
 * measurement. Each child runs in a private COPY of the working tree instead.
 *
 * A copy, deliberately, NOT a `git worktree`: a worktree checks out COMMITTED
 * state, so it would measure code WITHOUT the uncommitted / untracked changes a
 * study is usually iterating on — the wrong thing entirely. The copy preserves
 * the exact tree, and `--reflink=auto` makes it copy-on-write-cheap where the
 * filesystem supports it. Limitation: only relative-path writes are isolated; an
 * arm that writes a fixed ABSOLUTE path (e.g. /tmp/out) still collides, so keep
 * those per-invocation-unique (mktemp). BASI_STUDY_SHARE_CWD=1 opts out when the
 * arms are already write-isolated by construction. */
static bool study_isolation_on(void) {
    const char *e = getenv("BASI_STUDY_SHARE_CWD");
    return !(e && *e && atoi(e));            /* on by default; env opts out */
}

/* Copy `cwd`'s tree into `sandbox` (an existing empty dir) and chdir into it.
 * Returns true on success; on failure the child simply keeps the shared cwd. */
static bool study_sandbox_enter(const char *cwd, const char *sandbox) {
    char cmd[8192];
    /* `./.` copies dotfiles too; reflink where supported, else a plain copy. */
    snprintf(cmd, sizeof(cmd),
        "cp -a --reflink=auto '%s/.' '%s/' 2>/dev/null || cp -a '%s/.' '%s/' 2>/dev/null",
        cwd, sandbox, cwd, sandbox);
    if (system(cmd) != 0) return false;
    return chdir(sandbox) == 0;
}

/* Copy the studies this child produced back into the real cwd so they persist
 * for inspection, then delete the sandbox. Runs in the parent after the join. */
static void study_sandbox_leave(const char *sandbox, const char *cwd) {
    char cmd[8600];
    snprintf(cmd, sizeof(cmd),
        "mkdir -p '%s/%s' 2>/dev/null && cp -a '%s/%s/.' '%s/%s/' 2>/dev/null",
        cwd, KB_STUDIES_DIR, sandbox, KB_STUDIES_DIR, cwd, KB_STUDIES_DIR);
    (void) system(cmd);
    snprintf(cmd, sizeof(cmd), "rm -rf '%s'", sandbox);
    (void) system(cmd);
}

/* Concurrent trajectory dispatch — the fan-out the serial study_loop cannot do.
 *
 * study_loop runs its N trajectories one after another, on purpose: each is
 * handed a brief of what the earlier ones covered so it explores elsewhere.
 * That also serialises the LLM work — N trajectories take ~N times as long —
 * even though one llama-server can decode several sequences at once through its
 * parallel KV slots (`-np N --kv-unified`) and continuous batching.
 *
 * This variant forks one child per trajectory (in waves of STUDY_MAX_CONCURRENT)
 * so their /v1/chat/completions requests are in flight together and the server
 * batches them. fork() is the right primitive here, not threads:
 *   - retrieval memory is an in-process array (memory.c g_chunks); the
 *     copy-on-write fork gives each child its own view, so mem_add/mem_retrieve
 *     cannot race and no index can be corrupted.
 *   - the finds array and every StringBuf are per-child for free.
 *   - study artifacts do not collide because slugs are per-trajectory (-tN).
 * Each child ships its findings (a pointer-free POD) and its log back through a
 * private temp file, and the parent merges and runs the same ranking synthesis.
 * A temp file rather than a pipe because a multi-round log easily exceeds the
 * pipe buffer, which would deadlock a child before the parent starts draining.
 *
 * WHAT THIS TRADES, stated plainly:
 *   - Within one wave, trajectories are INDEPENDENT: they cannot see each
 *     other's ground, so the serial path's "push elsewhere" diversity is only
 *     enforced ACROSS waves (each wave is handed the prior waves' brief).
 *   - Arm commands execute in a PRIVATE COPY of the working tree per child
 *     (study_sandbox_enter), so relative-path writes no longer collide and the
 *     studies are copied back afterward. Two sharp edges remain: an arm writing a
 *     fixed ABSOLUTE path still collides (keep those mktemp-unique), and the copy
 *     costs one working-tree copy per child (reflink-cheap where supported;
 *     BASI_STUDY_SHARE_CWD=1 opts out when arms are already isolated).
 *   - id_slot is left unset, so the server AUTO-ASSIGNS an idle slot per request.
 *     Continuous batching load-balances better than static pinning, and a
 *     crashed child cannot strand a pinned slot. To pin instead, set
 *     req["id_slot"] in srvchat.cpp's build_request.
 */
static char *study_loop_concurrent(const char *seed_slug, const StudyLoopOpts *opts) {
    int ntraj = opts->trajectories > 0 ? opts->trajectories : 1;

    /* Spawn the retrieval embedder ONCE, here in the parent, so every child
     * shares that single embedder server over HTTP. Without it, each child's
     * first mem_retrieve would srvgen_spawn its own embedder — N model loads and
     * a fight over the port. Best-effort: retrieval degrades gracefully if the
     * embedder cannot start. */
    (void) embed_init();

    /* Working directory each child isolates a copy of. A path with a single quote
     * would break the shell copy commands, so fall back to sharing it (rare). */
    char cwdbuf[4096];
    if (!getcwd(cwdbuf, sizeof(cwdbuf))) cwdbuf[0] = '\0';
    bool isolate = study_isolation_on() && cwdbuf[0] && !strchr(cwdbuf, '\'');

    StringBuf log, explored;
    sb_init(&log);
    sb_init(&explored);      /* carried across waves, not within one */
    TrajFinding finds[20];
    int nfind = 0;
    const int maxfind = (int)(sizeof(finds) / sizeof(finds[0]));

    {
        char hdr[512];
        snprintf(hdr, sizeof(hdr),
            "\n████ dispatching %d trajectories, up to %d concurrent ████\n"
            "The server needs `-np %d --kv-unified` or the requests serialise.\n"
            "Arms run in %s.\n"
            "Live progress below interleaves across trajectories.\n",
            ntraj, STUDY_MAX_CONCURRENT,
            ntraj < STUDY_MAX_CONCURRENT ? ntraj : STUDY_MAX_CONCURRENT,
            isolate ? "a private copy of the working tree per trajectory"
                    : "the SHARED working directory (BASI_STUDY_SHARE_CWD) — "
                      "arms must be write-isolated by construction");
        sb_append_str(&log, hdr);
        fputs(hdr, stderr);
    }

    for (int base = 0; base < ntraj; base += STUDY_MAX_CONCURRENT) {
        int wave = ntraj - base;
        if (wave > STUDY_MAX_CONCURRENT) wave = STUDY_MAX_CONCURRENT;

        pid_t pids[STUDY_MAX_CONCURRENT];
        char  paths[STUDY_MAX_CONCURRENT][64];
        char *sandboxes[STUDY_MAX_CONCURRENT] = {0};   /* per-child tree copy, or NULL */
        /* Snapshot of what earlier waves covered — every child in this wave gets
         * the same brief, so diversity is enforced wave-to-wave. */
        char *wave_brief = explored.len ? strdup(explored.data) : NULL;

        for (int k = 0; k < wave; k++) {
            int t = base + k + 1;
            pids[k] = -1;
            paths[k][0] = '\0';

            /* Create the result file in the PARENT so its path is known here; the
             * forked child inherits nothing but the name and opens it fresh. */
            snprintf(paths[k], sizeof(paths[k]), "/tmp/basi_study_traj_XXXXXX");
            int fd = mkstemp(paths[k]);
            if (fd < 0) { paths[k][0] = '\0'; continue; }
            close(fd);

            /* Empty sandbox dir, created in the PARENT so the path is known here
             * for cleanup; the child fills it (concurrently) and chdir's in. */
            if (isolate) {
                char tmpl[] = "/tmp/basi-traj-XXXXXX";
                char *d = mkdtemp(tmpl);
                if (d) sandboxes[k] = strdup(d);
            }

            char tslug[192];
            snprintf(tslug, sizeof(tslug), "%.150s-t%d", seed_slug, t);

            pids[k] = fork();
            if (pids[k] == 0) {
                /* ── child: one independent trajectory ── */
                if (sandboxes[k]) study_sandbox_enter(cwdbuf, sandboxes[k]);
                TrajFinding cf[20];
                int cn = 0;
                StringBuf cfind;
                sb_init(&cfind);
                char *clog = study_trajectory(tslug, opts, wave_brief,
                                              STUDY_ANGLES[(t - 1) % STUDY_NUM_ANGLES],
                                              &cfind, NULL,
                                              cf, &cn, (int)(sizeof(cf)/sizeof(cf[0])));
                FILE *rf = fopen(paths[k], "wb");
                if (rf) {
                    /* frame: [int cn][cn × TrajFinding][int fl][fl brief bytes][log…] */
                    fwrite(&cn, sizeof(cn), 1, rf);
                    if (cn > 0) fwrite(cf, sizeof(TrajFinding), (size_t)cn, rf);
                    int fl = (int)cfind.len;
                    fwrite(&fl, sizeof(fl), 1, rf);
                    if (fl > 0) fwrite(cfind.data, 1, (size_t)fl, rf);
                    if (clog && *clog) fwrite(clog, 1, strlen(clog), rf);
                    fclose(rf);
                }
                _exit(0);   /* _exit, not exit: never run atexit — the embedder is shared */
            }
            if (pids[k] < 0) { unlink(paths[k]); paths[k][0] = '\0'; }
        }
        free(wave_brief);

        /* ── parent: join and merge, in trajectory order ── */
        for (int k = 0; k < wave; k++) {
            if (pids[k] > 0) {
                waitpid(pids[k], NULL, 0);
                FILE *rf = paths[k][0] ? fopen(paths[k], "rb") : NULL;
                if (rf) {
                    int cn = 0;
                    if (fread(&cn, sizeof(cn), 1, rf) == 1 && cn >= 0) {
                        for (int i = 0; i < cn; i++) {
                            TrajFinding tf;
                            if (fread(&tf, sizeof(tf), 1, rf) != 1) break;
                            if (nfind < maxfind) finds[nfind++] = tf;
                        }
                        int fl = 0;
                        if (fread(&fl, sizeof(fl), 1, rf) == 1 && fl > 0) {
                            char *fb = malloc((size_t)fl);
                            if (fb && fread(fb, 1, (size_t)fl, rf) == (size_t)fl)
                                sb_append(&explored, fb, (size_t)fl);
                            free(fb);
                        }
                        char buf[4096];
                        size_t r;
                        while ((r = fread(buf, 1, sizeof(buf), rf)) > 0) sb_append(&log, buf, r);
                    }
                    fclose(rf);
                }
            }
            if (paths[k][0]) unlink(paths[k]);
            /* Copy this child's studies back to the real cwd, then drop the copy. */
            if (sandboxes[k]) {
                study_sandbox_leave(sandboxes[k], cwdbuf);
                free(sandboxes[k]);
                sandboxes[k] = NULL;
            }
        }
    }

    study_rank_synthesis(&log, finds, nfind);
    sb_free(&explored);
    sb_append_str(&log, "\nloop finished.\n");
    return sb_to_str(&log);
}

char *study_loop(const char *seed_slug, const StudyLoopOpts *opts) {
    int ntraj = opts->trajectories > 0 ? opts->trajectories : 1;
    if (opts->concurrent && ntraj > 1)
        return study_loop_concurrent(seed_slug, opts);
    StringBuf log, explored;
    sb_init(&log);
    sb_init(&explored);
    char line[512];
    TrajFinding finds[20];
    int nfind = 0;

    for (int t = 1; t <= ntraj; t++) {
        char tslug[192];
        if (ntraj > 1) snprintf(tslug, sizeof(tslug), "%.150s-t%d", seed_slug, t);
        else           snprintf(tslug, sizeof(tslug), "%.180s", seed_slug);

        if (ntraj > 1) {
            snprintf(line, sizeof(line), "\n████ trajectory %d/%d ████\n", t, ntraj);
            sb_append_str(&log, line);
            fputs(line, stderr);
        }

        StringBuf finding;
        sb_init(&finding);
        char *done = NULL;
        char *sub = study_trajectory(tslug, opts,
                                     explored.len ? explored.data : NULL,
                                     STUDY_ANGLES[(t - 1) % STUDY_NUM_ANGLES],
                                     &finding, &done,
                                     finds, &nfind, (int)(sizeof(finds)/sizeof(finds[0])));
        sb_append_str(&log, sub ? sub : "");
        free(sub);
        free(done);

        if (finding.len) {
            sb_append(&explored, finding.data, finding.len);
            sb_free(&finding);
        } else {
            sb_free(&finding);
        }
    }

    if (ntraj > 1 && explored.len) {
        sb_append_str(&log, "\n════ what every trajectory covered ════\n\n");
        sb_append(&log, explored.data, explored.len);
    }

    /* Ranked synthesis. ROBIN orders its candidates by Bradley-Terry-Luce over
     * model preference; these carry measured effect sizes and p-values, so they
     * are ordered by those instead — supported findings first, largest effect
     * first. Nothing here is a judgement call. */
    study_rank_synthesis(&log, finds, nfind);
    sb_free(&explored);
    sb_append_str(&log, "\nloop finished.\n");
    return sb_to_str(&log);
}

/* ── CLI ───────────────────────────────────────────────────────────────── */

static void cli_progress(const char *arm, int run, int of, void *ud) {
    (void)ud;
    fprintf(stderr, "  arm %s: run %d/%d done\n", arm, run, of);
}

static int study_cmd_list(void) {
    DIR *d = opendir(KB_STUDIES_DIR);
    if (!d) { printf("No studies yet (%s does not exist).\n", KB_STUDIES_DIR); return 0; }
    struct dirent *e;
    int n = 0;
    while ((e = readdir(d)) != NULL) {
        size_t l = strlen(e->d_name);
        if (l < 4 || strcmp(e->d_name + l - 3, ".md") != 0) continue;
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", KB_STUDIES_DIR, e->d_name);
        size_t len = 0;
        char *c = kb_read_file(path, &len);
        if (!c) continue;
        KbFrontmatter fm;
        const char *status = "?";
        if (kb_parse_frontmatter(c, len, &fm) >= 0) {
            const char *s = kb_fm_get(&fm, "status");
            if (s) status = s;
            printf("  %-40s %s\n", e->d_name, status);
            kb_fm_free(&fm);
        }
        free(c);
        n++;
    }
    closedir(d);
    if (!n) printf("No studies yet.\n");
    return 0;
}

int cmd_study(int argc, char **argv) {
    if (argc < 1) {
        fprintf(stderr,
            "usage: basi-cli study <run|list|show> [slug]\n"
            "  run <slug>    execute the study's arms and compute the verdict\n"
            "  loop <slug>   run rounds unattended: verdict -> next hypothesis\n"
            "                --question \"...\" --allow \"<prefixes>\" designs the first\n"
            "                study too, so any question can be run end to end\n"
            "                --trajectories N runs N INDEPENDENT explorations, each told\n"
            "                what the others covered so it probes a different dimension\n"
            "                --concurrent dispatches those trajectories at once against one\n"
            "                server (needs -np N --kv-unified); see study_loop_concurrent\n"
            "  list         list studies and their status\n"
            "  show <slug>  print the study artifact\n");
        return 2;
    }
    if (strcmp(argv[0], "list") == 0) return study_cmd_list();

    if (strcmp(argv[0], "show") == 0) {
        if (argc < 2) { fprintf(stderr, "study show: missing slug\n"); return 2; }
        char *path = study_path(argv[1]);
        if (!path) return 1;
        size_t len = 0;
        char *c = kb_read_file(path, &len);
        if (!c) { fprintf(stderr, "cannot read %s\n", path); free(path); return 1; }
        fwrite(c, 1, len, stdout);
        free(c); free(path);
        return 0;
    }

    if (strcmp(argv[0], "loop") == 0) {
        if (argc < 2) { fprintf(stderr, "study loop: missing seed slug\n"); return 2; }
        StudyLoopOpts o = { .max_rounds = 5, .port = 8181, .unsafe = false,
                            .question = NULL, .allow = NULL, .trajectories = 1,
                            .concurrent = false };
        const char *pe = getenv("BASI_SERVER_PORT");
        if (pe && *pe) o.port = atoi(pe);
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--max-rounds") == 0 && i + 1 < argc) o.max_rounds = atoi(argv[++i]);
            else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc)  o.port = atoi(argv[++i]);
            else if (strcmp(argv[i], "--question") == 0 && i + 1 < argc) o.question = argv[++i];
            else if (strcmp(argv[i], "--allow") == 0 && i + 1 < argc)  o.allow = argv[++i];
            else if (strcmp(argv[i], "--trajectories") == 0 && i + 1 < argc) o.trajectories = atoi(argv[++i]);
            else if (strcmp(argv[i], "--concurrent") == 0)            o.concurrent = true;
            else if (strcmp(argv[i], "--unsafe") == 0)                o.unsafe = true;
            else { fprintf(stderr, "study loop: unknown option '%s'\n", argv[i]); return 2; }
        }
        if (o.trajectories < 1 || o.trajectories > 20) {
            fprintf(stderr, "study loop: --trajectories must be 1..20\n");
            return 2;
        }
        if (o.max_rounds < 0 || o.max_rounds > 1000) {
            fprintf(stderr, "study loop: --max-rounds must be 0..1000 (0 = until STOP)\n");
            return 2;
        }
        char hc[192];
        snprintf(hc, sizeof(hc),
                 "curl -sf -o /dev/null http://127.0.0.1:%d/health", o.port);
        if (system(hc) != 0) {
            fprintf(stderr,
                "study loop: no healthy llama-server on 127.0.0.1:%d.\n"
                "The loop needs one to propose hypotheses; start it, or pass --port.\n",
                o.port);
            return 1;
        }
        char *rep = study_loop(argv[1], &o);
        fputs(rep, stdout);
        free(rep);
        return 0;
    }

    if (strcmp(argv[0], "run") == 0) {
        if (argc < 2) { fprintf(stderr, "study run: missing slug\n"); return 2; }
        char *rep = study_run_slug(argv[1], cli_progress, NULL);
        fputs(rep, stdout);
        bool bad = strstr(rep, "INCONCLUSIVE") || strstr(rep, "invalid") || strstr(rep, "cannot read");
        free(rep);
        return bad ? 1 : 0;
    }

    fprintf(stderr, "study: unknown subcommand '%s'\n", argv[0]);
    return 2;
}
