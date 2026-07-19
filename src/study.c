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

#include "util.h"
#include "kb.h"
#include "study.h"

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
    for (int i = 0; i < s->narms; i++) { free(s->arms[i].name); free(s->arms[i].command); }
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
static bool extract_metric(const regex_t *re, const char *text, double *out) {
    regmatch_t m[2];
    const char *cur = text;
    bool found = false;
    double last = 0;
    int flags = 0;
    while (*cur && regexec(re, cur, 2, m, flags) == 0) {
        if (m[1].rm_so >= 0) {
            size_t len = (size_t)(m[1].rm_eo - m[1].rm_so);
            char buf[64];
            if (len < sizeof(buf)) {
                memcpy(buf, cur + m[1].rm_so, len);
                buf[len] = '\0';
                char *end = NULL;
                double v = strtod(buf, &end);
                if (end != buf) { last = v; found = true; }
            }
        }
        regoff_t adv = (m[0].rm_eo > m[0].rm_so) ? m[0].rm_eo : m[0].rm_so + 1;
        cur += adv;
        flags = REG_NOTBOL;
    }
    if (found) *out = last;
    return found;
}

void study_execute(Study *s, StudyProgressFn progress, void *ud) {
    regex_t re;
    if (regcomp(&re, s->extract, REG_EXTENDED) != 0) return;  /* validated already */

    for (int a = 0; a < s->narms; a++) {
        StudyArm *arm = &s->arms[a];
        arm->nruns = 0;
        for (int i = 0; i < s->runs && i < STUDY_MAX_RUNS; i++) {
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
                } else if (extract_metric(&re, outp, &r->value)) {
                    r->status = RUN_OK;
                } else {
                    r->status = RUN_NO_METRIC;
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

/* Rewrite only the `status:` line inside the frontmatter block. */
static char *replace_status(const char *content, size_t len, const char *new_status) {
    const char *p = content;
    const char *end = content + len;
    if (len < 4 || strncmp(p, "---", 3) != 0) return NULL;
    p += 3;
    while (p < end && *p != '\n') p++;
    if (p < end) p++;

    StringBuf b;
    sb_init(&b);
    sb_append(&b, content, (size_t)(p - content));
    bool wrote = false;
    while (p < end) {
        const char *le = p;
        while (le < end && *le != '\n') le++;
        size_t llen = (size_t)(le - p);
        if (llen >= 3 && strncmp(p, "---", 3) == 0) { sb_append(&b, p, (size_t)(end - p)); break; }
        if (!wrote && llen > 7 && strncmp(p, "status:", 7) == 0) {
            sb_append_str(&b, "status: ");
            sb_append_str(&b, new_status);
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
    const char *slug = kb_fm_get(&fm, "slug");
    if (!slug) { kb_fm_free(&fm); return strdup("study_write internal: validator passed but slug missing"); }

    if (mkdir_p(KB_STUDIES_DIR) != 0) {
        char *m = malloc(256);
        snprintf(m, 256, "study_write failed: cannot create %s (%s)\n", KB_STUDIES_DIR, strerror(errno));
        kb_fm_free(&fm);
        return m;
    }

    char *dest = study_path(slug);
    if (!dest) { kb_fm_free(&fm); return strdup("study_write failed: destination path too long"); }

    FILE *f = fopen(dest, "w");
    if (!f) {
        char *m = malloc(512);
        snprintf(m, 512, "study_write failed: cannot write %s (%s)\n", dest, strerror(errno));
        kb_fm_free(&fm); free(dest);
        return m;
    }
    fwrite(body, 1, body_len, f);
    if (body_len == 0 || body[body_len - 1] != '\n') fputc('\n', f);
    fclose(f);
    kb_fm_free(&fm);

    char *msg = malloc(640);
    snprintf(msg, 640,
        "Study written to %s. Validation passed (frontmatter, sections, arms, "
        "extract regex, decision rule).\nRun it with: basi study run %s\n"
        "The verdict will be computed from the command output, not from your reading of it.\n",
        dest, slug);
    free(dest);
    return msg;
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
            "  run <slug>   execute the study's arms and compute the verdict\n"
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
