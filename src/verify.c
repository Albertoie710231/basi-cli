#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/wait.h>
#include <errno.h>

#include "globals.h"
#include "util.h"
#include "kb.h"
#include "verify.h"

/* ── Markdown table parser (Implementation Plan only) ──────────────── */

#define VERIFY_MAX_COLS    16
#define VERIFY_TIMEOUT_S   30   /* per-clause wall-clock cap */
#define VERIFY_TAIL_LINES   5
#define VERIFY_TAIL_BYTES  640

typedef struct {
    char *cells[VERIFY_MAX_COLS];
    int   n;
} Row;

static void row_free(Row *r) {
    for (int i = 0; i < r->n; i++) free(r->cells[i]);
    r->n = 0;
}

/* Trim ASCII whitespace + surrounding backticks. Returns malloc'd string. */
static char *trim_cell(const char *s, size_t len) {
    while (len && (s[0] == ' ' || s[0] == '\t')) { s++; len--; }
    while (len && (s[len - 1] == ' ' || s[len - 1] == '\t' ||
                   s[len - 1] == '\r')) len--;
    if (len >= 2 && s[0] == '`' && s[len - 1] == '`') { s++; len -= 2; }
    while (len && (s[0] == ' ' || s[0] == '\t')) { s++; len--; }
    while (len && (s[len - 1] == ' ' || s[len - 1] == '\t')) len--;
    char *out = malloc(len + 1);
    memcpy(out, s, len);
    out[len] = '\0';
    return out;
}

/* Splits a `|`-delimited row. Strips leading/trailing | and per-cell
 * whitespace + backticks. Treats `\|` as a literal | within a cell. */
static void split_row(const char *line, size_t len, Row *out) {
    out->n = 0;
    /* Skip leading whitespace + opening |. */
    size_t i = 0;
    while (i < len && (line[i] == ' ' || line[i] == '\t')) i++;
    if (i < len && line[i] == '|') i++;
    /* Trim trailing whitespace + closing |. */
    size_t end = len;
    while (end > i && (line[end - 1] == ' ' || line[end - 1] == '\t' ||
                       line[end - 1] == '\r')) end--;
    if (end > i && line[end - 1] == '|') end--;

    StringBuf cell;
    sb_init(&cell);
    for (size_t k = i; k < end; k++) {
        if (line[k] == '\\' && k + 1 < end && line[k + 1] == '|') {
            sb_append_char(&cell, '|');
            k++;
        } else if (line[k] == '|') {
            if (out->n < VERIFY_MAX_COLS) {
                out->cells[out->n++] = trim_cell(cell.data, cell.len);
            }
            sb_clear(&cell);
        } else {
            sb_append_char(&cell, line[k]);
        }
    }
    if (out->n < VERIFY_MAX_COLS) {
        out->cells[out->n++] = trim_cell(cell.data, cell.len);
    }
    sb_free(&cell);
}

/* Returns true if the row is the divider line (cells consist of only
 * dashes / colons / spaces). */
static bool row_is_divider(const Row *r) {
    if (r->n == 0) return false;
    for (int i = 0; i < r->n; i++) {
        for (const char *p = r->cells[i]; *p; p++) {
            if (*p != '-' && *p != ':' && *p != ' ' && *p != '\t') return false;
        }
    }
    return true;
}

static int find_col(const Row *header, const char *name) {
    for (int i = 0; i < header->n; i++) {
        if (strcmp(header->cells[i], name) == 0) return i;
    }
    return -1;
}

/* ── Run one verify clause ────────────────────────────────────────── */

typedef enum {
    VRES_OK,
    VRES_FAIL,
    VRES_SETUP,    /* exit 127 or signal */
    VRES_SKIP      /* empty / "-" / "—" */
} VerifyKind;

typedef struct {
    VerifyKind kind;
    int        exit_code;
    char      *tail;       /* malloc'd; last few lines of combined output */
} VerifyResult;

static bool clause_is_skip(const char *clause) {
    if (!clause || !*clause) return true;
    /* common dash variants people type for "no verify" */
    if (strcmp(clause, "-") == 0)        return true;
    if (strcmp(clause, "—") == 0)        return true;
    if (strcmp(clause, "...") == 0)      return true;
    if (strcmp(clause, "…") == 0)        return true;
    if (strcmp(clause, "n/a") == 0)      return true;
    if (strcmp(clause, "N/A") == 0)      return true;
    return false;
}

/* Wraps clause as `timeout Ns bash -c '<clause>' 2>&1` and runs via popen.
 * Captures up to VERIFY_TAIL_BYTES of combined output; keeps only the last
 * VERIFY_TAIL_LINES lines for the report. Sets *exit_code to the wait
 * status interpretation (127 = setup, signal-killed = -1). */
static VerifyResult run_clause(const char *clause) {
    VerifyResult r = { VRES_SKIP, 0, NULL };
    if (clause_is_skip(clause)) return r;

    StringBuf wrapped;
    sb_init(&wrapped);
    char to[32];
    snprintf(to, sizeof(to), "timeout %ds bash -c '", VERIFY_TIMEOUT_S);
    sb_append_str(&wrapped, to);
    for (const char *c = clause; *c; c++) {
        if (*c == '\'') sb_append_str(&wrapped, "'\"'\"'");
        else sb_append_char(&wrapped, *c);
    }
    sb_append_str(&wrapped, "' 2>&1");

    /* Shared popen→capped-read→drain→pclose loop; gives us the wait-status
     * decoded exit code (127 = setup, negative = signal). */
    int exit_code = -1;
    char *out = run_command_status(sb_to_str(&wrapped), VERIFY_TAIL_BYTES, &exit_code);
    sb_free(&wrapped);
    r.exit_code = exit_code;

    /* Keep only the last VERIFY_TAIL_LINES lines. */
    const char *p = out;
    size_t total = strlen(out);
    if (total > 0) {
        int newlines = 0;
        size_t cut = total;
        for (size_t i = total; i > 0; i--) {
            if (p[i - 1] == '\n') {
                newlines++;
                if (newlines > VERIFY_TAIL_LINES) { cut = i; break; }
            }
        }
        const char *start = p + cut;
        size_t slen = total - cut;
        while (slen && (*start == '\n' || *start == '\r')) { start++; slen--; }
        r.tail = malloc(slen + 1);
        memcpy(r.tail, start, slen);
        r.tail[slen] = '\0';
    } else {
        r.tail = strdup("");
    }
    free(out);

    if (r.exit_code == 0)        r.kind = VRES_OK;
    else if (r.exit_code == 127) r.kind = VRES_SETUP;
    else if (r.exit_code < 0)    r.kind = VRES_SETUP;   /* signal */
    else                         r.kind = VRES_FAIL;
    return r;
}

/* ── Tool entry ───────────────────────────────────────────────────── */

static const char *find_impl_plan(const char *body, size_t body_len, size_t *out_len) {
    const char *needle = "## Implementation Plan";
    size_t nlen = strlen(needle);
    const char *p = body, *end = body + body_len;
    while (p < end) {
        const char *le = p;
        while (le < end && *le != '\n') le++;
        if ((size_t)(le - p) >= nlen && strncmp(p, needle, nlen) == 0) {
            const char *start = (le < end) ? le + 1 : end;
            const char *q = start;
            while (q < end) {
                const char *qe = q;
                while (qe < end && *qe != '\n') qe++;
                if (qe - q >= 3 && q[0] == '#' && q[1] == '#' && q[2] == ' ') {
                    *out_len = (size_t)(q - start);
                    return start;
                }
                q = (qe < end) ? qe + 1 : end;
            }
            *out_len = (size_t)(end - start);
            return start;
        }
        p = (le < end) ? le + 1 : end;
    }
    return NULL;
}

char *execute_plan_verify(const char *args) {
    while (args && (*args == ' ' || *args == '\t' || *args == '\n')) args++;
    char id_filter[64] = "";
    if (args && *args) {
        size_t i = 0;
        while (args[i] && args[i] != ' ' && args[i] != '\t' && args[i] != '\n' &&
               i + 1 < sizeof(id_filter)) {
            id_filter[i] = args[i];
            i++;
        }
        id_filter[i] = '\0';
    }

    if (!current_plan_slug) {
        return strdup("plan_verify failed: no current plan slug. /plan <slug> first.");
    }
    char path[1024];
    if ((size_t)snprintf(path, sizeof(path), "%s/%s.md",
                         KB_PLANS_DIR, current_plan_slug) >= sizeof(path)) {
        return strdup("plan_verify failed: plan path too long.");
    }
    size_t flen = 0;
    char *file = kb_read_file(path, &flen);
    if (!file) {
        char *msg = malloc(512);
        snprintf(msg, 512, "plan_verify failed: cannot read %.300s (%s).\n",
                 path, strerror(errno));
        return msg;
    }

    size_t section_len = 0;
    const char *section = find_impl_plan(file, flen, &section_len);
    if (!section) {
        free(file);
        return strdup("plan_verify failed: ## Implementation Plan section not found.");
    }

    /* Parse rows from the section. First non-empty `|`-row is header,
     * second is divider, rest are data. */
    Row header = { {0}, 0 };
    int verify_col = -1, id_col = -1, title_col = -1;
    bool got_header = false, got_divider = false;

    StringBuf out;
    sb_init(&out);
    sb_append_str(&out, "Verify results for ");
    sb_append_str(&out, current_plan_slug);
    if (id_filter[0]) {
        sb_append_str(&out, " (id=");
        sb_append_str(&out, id_filter);
        sb_append_str(&out, ")");
    }
    sb_append_str(&out, ":\n\n");

    int n_ok = 0, n_fail = 0, n_setup = 0, n_skip = 0, n_total = 0;
    bool matched_filter = false;

    const char *p = section;
    const char *end = section + section_len;
    while (p < end) {
        const char *le = p;
        while (le < end && *le != '\n') le++;
        size_t llen = (size_t)(le - p);
        const char *trim = p;
        size_t tlen = llen;
        while (tlen && (*trim == ' ' || *trim == '\t')) { trim++; tlen--; }
        if (tlen > 0 && *trim == '|') {
            Row r = { {0}, 0 };
            split_row(p, llen, &r);
            if (!got_header) {
                header = r;
                got_header = true;
                verify_col = find_col(&header, "verify");
                id_col     = find_col(&header, "id");
                title_col  = find_col(&header, "title");
                if (verify_col < 0 || id_col < 0) {
                    sb_append_str(&out,
                        "ERROR: Implementation Plan table missing 'id' or "
                        "'verify' column.\n");
                    row_free(&header);
                    free(file);
                    return sb_to_str(&out);
                }
            } else if (!got_divider) {
                if (row_is_divider(&r)) got_divider = true;
                row_free(&r);
            } else {
                /* data row */
                if (id_col < r.n && verify_col < r.n) {
                    const char *id     = r.cells[id_col];
                    const char *title  = (title_col >= 0 && title_col < r.n)
                                         ? r.cells[title_col] : "";
                    const char *clause = r.cells[verify_col];
                    if (id_filter[0] && strcmp(id_filter, id) != 0) {
                        row_free(&r);
                        p = (le < end) ? le + 1 : end;
                        continue;
                    }
                    matched_filter = matched_filter || (id_filter[0] != '\0');
                    n_total++;
                    VerifyResult vr = run_clause(clause);
                    const char *tag =
                        vr.kind == VRES_OK    ? "OK   " :
                        vr.kind == VRES_FAIL  ? "FAIL " :
                        vr.kind == VRES_SETUP ? "SETUP" :
                                                "SKIP ";
                    if (vr.kind == VRES_OK)         n_ok++;
                    else if (vr.kind == VRES_FAIL)  n_fail++;
                    else if (vr.kind == VRES_SETUP) n_setup++;
                    else                            n_skip++;

                    char header_line[256];
                    snprintf(header_line, sizeof(header_line),
                        "%s  %s  %.80s", tag, id, title);
                    sb_append_str(&out, header_line);
                    sb_append_char(&out, '\n');

                    if (vr.kind == VRES_FAIL) {
                        char xs[64];
                        snprintf(xs, sizeof(xs), "       exit %d\n", vr.exit_code);
                        sb_append_str(&out, xs);
                    } else if (vr.kind == VRES_SETUP) {
                        char xs[96];
                        if (vr.exit_code == 127) {
                            snprintf(xs, sizeof(xs),
                                "       exit 127 — command not found / setup failure\n");
                        } else if (vr.exit_code < 0) {
                            snprintf(xs, sizeof(xs),
                                "       killed by signal %d\n", -vr.exit_code);
                        } else {
                            snprintf(xs, sizeof(xs),
                                "       exit %d (treated as setup)\n", vr.exit_code);
                        }
                        sb_append_str(&out, xs);
                    } else if (vr.kind == VRES_SKIP) {
                        sb_append_str(&out,
                            "       (verify clause empty / placeholder — write a runnable check)\n");
                    }
                    if (vr.tail && vr.tail[0] && vr.kind != VRES_OK && vr.kind != VRES_SKIP) {
                        const char *tp = vr.tail;
                        while (*tp) {
                            const char *tle = tp;
                            while (*tle && *tle != '\n') tle++;
                            sb_append_str(&out, "       │ ");
                            sb_append(&out, tp, (size_t)(tle - tp));
                            sb_append_char(&out, '\n');
                            if (!*tle) break;
                            tp = tle + 1;
                        }
                    }
                    free(vr.tail);
                }
                row_free(&r);
            }
        }
        p = (le < end) ? le + 1 : end;
    }
    row_free(&header);
    free(file);

    if (n_total == 0) {
        if (id_filter[0]) {
            char *msg = malloc(256);
            snprintf(msg, 256,
                "plan_verify: no row with id='%s' in the Implementation Plan table.\n",
                id_filter);
            sb_free(&out);
            return msg;
        }
        sb_append_str(&out,
            "(no data rows found — Implementation Plan table is empty or malformed)\n");
        return sb_to_str(&out);
    }
    (void)matched_filter;

    char summary[160];
    snprintf(summary, sizeof(summary),
        "\nSummary: %d OK, %d FAIL, %d SETUP, %d SKIP — %d row%s.\n",
        n_ok, n_fail, n_setup, n_skip, n_total, n_total == 1 ? "" : "s");
    sb_append_str(&out, summary);
    return sb_to_str(&out);
}
