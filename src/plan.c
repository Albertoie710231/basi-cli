#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>

#include "util.h"
#include "kb.h"
#include "plan.h"

/* ── Plan artifact validator + plan_write tool ────────────────────── */

#define PLAN_MAX_LINES 200

static const char *PLAN_REQUIRED_FRONTMATTER_KEYS[] = {
    "slug", "status", "created", "goal", NULL
};

static const char *PLAN_VALID_STATUSES[] = {
    "drafting", "spike", "premortem", "active", "done", "abandoned", NULL
};

/* The seven Proposal-A3 sections, in order. */
static const char *A3_SECTIONS[] = {
    "Theme", "Background", "Current Condition", "Cause Analysis",
    "Target Condition", "Implementation Plan", "Follow-Up", NULL
};

/* Case-insensitive equality for an arbitrary-length slice vs. a C string. */
static bool kb_icaseq(const char *a, size_t alen, const char *b) {
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

/* True if `body` contains a "## <name>" heading line (case-insensitive). */
static bool plan_has_h2(const char *body, size_t body_len, const char *name) {
    const char *p = body;
    const char *end = body + body_len;
    while (p < end) {
        const char *le = p;
        while (le < end && *le != '\n') le++;
        if (le > p + 2 && p[0] == '#' && p[1] == '#' && p[2] == ' ') {
            const char *h = p + 3;
            while (h < le && (*h == ' ' || *h == '\t')) h++;
            const char *he = le;
            while (he > h && (he[-1] == ' ' || he[-1] == '\t' ||
                              he[-1] == '#' || he[-1] == '\r')) he--;
            if (kb_icaseq(h, he - h, name)) return true;
        }
        p = (le < end) ? le + 1 : end;
    }
    return false;
}

/* Returns NULL if the artifact is valid, otherwise a malloc'd, human-readable
 * description of all validation failures (one per line). */
char *validate_plan_artifact(const char *content, size_t content_len) {
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

    for (const char **k = PLAN_REQUIRED_FRONTMATTER_KEYS; *k; k++) {
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
        for (const char **s = PLAN_VALID_STATUSES; *s; s++) {
            if (strcmp(*s, status) == 0) { ok = true; break; }
        }
        if (!ok) {
            sb_append_str(&errs, "- status '");
            sb_append_str(&errs, status);
            sb_append_str(&errs,
                "' invalid; must be one of: drafting | spike | premortem | active | done | abandoned\n");
        }
    }

    const char *body = content + fm.body_offset;
    size_t body_len = content_len - fm.body_offset;
    for (const char **sec = A3_SECTIONS; *sec; sec++) {
        if (!plan_has_h2(body, body_len, *sec)) {
            sb_append_str(&errs, "- missing required A3 section: ## ");
            sb_append_str(&errs, *sec);
            sb_append_char(&errs, '\n');
        }
    }

    size_t lines = 1;
    for (size_t i = 0; i < content_len; i++) if (content[i] == '\n') lines++;
    if (lines > PLAN_MAX_LINES) {
        char buf[128];
        snprintf(buf, sizeof(buf),
            "- artifact is %zu lines (max %d) — compress, don't truncate\n",
            lines, PLAN_MAX_LINES);
        sb_append_str(&errs, buf);
    }

    kb_fm_free(&fm);
    if (errs.len == 0) {
        sb_free(&errs);
        return NULL;
    }
    return sb_to_str(&errs);
}

char *execute_plan_write(const char *body) {
    while (*body == ' ' || *body == '\t' || *body == '\n') body++;
    if (!*body) {
        return strdup("plan_write not allowed: missing plan body. See system prompt for the A3 template.");
    }
    size_t body_len = strlen(body);

    char *errs = validate_plan_artifact(body, body_len);
    if (errs) {
        StringBuf out;
        sb_init(&out);
        sb_append_str(&out, "plan_write rejected — fix these and resubmit:\n");
        sb_append_str(&out, errs);
        free(errs);
        return sb_to_str(&out);
    }

    KbFrontmatter fm;
    kb_parse_frontmatter(body, body_len, &fm);
    const char *slug = kb_fm_get(&fm, "slug");
    if (!slug) {
        kb_fm_free(&fm);
        return strdup("plan_write internal: validator passed but slug missing");
    }

    /* Runtime-phase agreement: the file's status must match the phase the
     * tool was actually called from. plan_tool_allowed already rejects
     * plan_write outside drafting/premortem, so those are the only valid
     * runtime phases here. */
    const char *claimed_status = kb_fm_get(&fm, "status");
    const char *expected_status =
        plan_phase == PHASE_DRAFTING  ? "drafting"  :
        plan_phase == PHASE_PREMORTEM ? "premortem" : NULL;
    if (claimed_status && expected_status &&
        strcmp(claimed_status, expected_status) != 0) {
        char *msg = malloc(384);
        snprintf(msg, 384,
            "plan_write rejected: frontmatter status='%s' but runtime phase is %s. "
            "Set status: %s in the frontmatter (file is source of truth).\n",
            claimed_status, plan_phase_name(plan_phase), expected_status);
        kb_fm_free(&fm);
        return msg;
    }

    if (current_plan_slug && strcmp(slug, current_plan_slug) != 0) {
        char *msg = malloc(384);
        snprintf(msg, 384,
            "plan_write rejected: slug '%s' does not match current plan '%s'.\n",
            slug, current_plan_slug);
        kb_fm_free(&fm);
        return msg;
    }

    if (kb_ensure_dirs() != 0) {
        kb_fm_free(&fm);
        char *msg = malloc(256);
        snprintf(msg, 256, "plan_write failed: cannot create %s (%s)\n",
                 KB_PLANS_DIR, strerror(errno));
        return msg;
    }

    char dest[1024];
    if ((size_t)snprintf(dest, sizeof(dest), "%s/%s.md", KB_PLANS_DIR, slug) >= sizeof(dest)) {
        kb_fm_free(&fm);
        return strdup("plan_write failed: destination path too long");
    }

    struct stat st;
    if (stat(dest, &st) == 0 && plan_phase != PHASE_PREMORTEM) {
        char *msg = malloc(512);
        snprintf(msg, 512,
            "plan_write rejected: %s already exists. Pick a different slug, or delete the old plan to start over.\n",
            dest);
        kb_fm_free(&fm);
        return msg;
    }

    FILE *f = fopen(dest, "w");
    if (!f) {
        char *msg = malloc(256);
        snprintf(msg, 256, "plan_write failed: cannot create %s (%s)\n", dest, strerror(errno));
        kb_fm_free(&fm);
        return msg;
    }
    fwrite(body, 1, body_len, f);
    if (body_len == 0 || body[body_len - 1] != '\n') fputc('\n', f);
    fclose(f);
    kb_fm_free(&fm);

    char *msg = malloc(512);
    snprintf(msg, 512,
        "Plan written to %s. Validation passed (frontmatter, A3 sections, %d-line cap).\n",
        dest, PLAN_MAX_LINES);
    return msg;
}

/* Rewrites only the `status:` line in the YAML frontmatter of
 * .basi/plans/<slug>.md. Operates on the closed range [start-of-file,
 * second '---' marker], so it cannot touch a body line that happens to
 * start with "status:". Atomic via tmpfile + rename. */
int rewrite_plan_status(const char *slug, const char *new_status) {
    char path[1024];
    if ((size_t)snprintf(path, sizeof(path), "%s/%s.md",
                         KB_PLANS_DIR, slug) >= sizeof(path)) {
        return -1;
    }
    size_t flen = 0;
    char *buf = kb_read_file(path, &flen);
    if (!buf) return -1;

    /* Find the closing --- of the frontmatter. */
    KbFrontmatter fm;
    int rc = kb_parse_frontmatter(buf, flen, &fm);
    if (rc < 0 || fm.body_offset == 0) {
        kb_fm_free(&fm);
        free(buf);
        return -2;
    }
    size_t fm_end = fm.body_offset;
    kb_fm_free(&fm);

    /* Locate the `status:` line within [0, fm_end). */
    size_t i = 0;
    size_t line_start = 0;
    bool found = false;
    while (i < fm_end) {
        line_start = i;
        while (i < fm_end && buf[i] != '\n') i++;
        size_t le = i;
        const char *p = buf + line_start;
        size_t plen = le - line_start;
        size_t off = 0;
        while (off < plen && (p[off] == ' ' || p[off] == '\t')) off++;
        if (plen - off >= 7 && strncmp(p + off, "status:", 7) == 0) {
            found = true;
            break;
        }
        if (i < fm_end) i++;
    }
    if (!found) {
        free(buf);
        return -2;
    }
    size_t line_end = line_start;
    while (line_end < fm_end && buf[line_end] != '\n') line_end++;

    char tmp[1280];
    if ((size_t)snprintf(tmp, sizeof(tmp), "%s.tmp", path) >= sizeof(tmp)) {
        free(buf);
        return -1;
    }
    FILE *f = fopen(tmp, "w");
    if (!f) {
        free(buf);
        return -1;
    }
    fwrite(buf, 1, line_start, f);
    fprintf(f, "status: %s", new_status);
    fwrite(buf + line_end, 1, flen - line_end, f);
    if (fclose(f) != 0) {
        unlink(tmp);
        free(buf);
        return -1;
    }
    if (rename(tmp, path) != 0) {
        unlink(tmp);
        free(buf);
        return -1;
    }
    free(buf);
    return 0;
}

/* ── Spike phase: assumption gate + spike artifact ─────────────────── */

#define SPIKE_MAX_LINES 80

static const char *SPIKE_REQUIRED_FRONTMATTER_KEYS[] = {
    "slug", "phase", "created", NULL
};

static const char *SPIKE_SECTIONS[] = {
    "Question", "Findings", "Decision", NULL
};

static const char *SPIKE_VALID_DECISIONS[] = {
    "PROCEED-TO-PLAN", "NEED-ANOTHER-SPIKE", "ABANDON", NULL
};

/* Counts list-item lines: starting with "- ", "* ", "<digits>. ", or "<digits>) "
 * after optional whitespace. */
static int count_list_items(const char *body, size_t body_len) {
    int count = 0;
    const char *p = body, *end = body + body_len;
    while (p < end) {
        const char *le = p;
        while (le < end && *le != '\n') le++;
        const char *t = p;
        while (t < le && (*t == ' ' || *t == '\t')) t++;
        if (t < le) {
            char c0 = t[0];
            char c1 = (t + 1 < le) ? t[1] : '\0';
            bool is_bullet = (c0 == '-' || c0 == '*') && (c1 == ' ' || c1 == '\t');
            bool is_num = false;
            if (c0 >= '0' && c0 <= '9') {
                const char *q = t;
                while (q < le && *q >= '0' && *q <= '9') q++;
                if (q < le && (*q == '.' || *q == ')') &&
                    q + 1 < le && (q[1] == ' ' || q[1] == '\t')) {
                    is_num = true;
                }
            }
            if (is_bullet || is_num) count++;
        }
        p = (le < end) ? le + 1 : end;
    }
    return count;
}

char *execute_assumptions(const char *body) {
    while (*body == ' ' || *body == '\t' || *body == '\n') body++;
    if (!*body) {
        return strdup(
            "assumptions not allowed: missing list. List unverified things, "
            "one per line, prefixed with '- ' or numbered. If three or more, "
            "you'll be routed to a spike phase to research before drafting.");
    }
    int n = count_list_items(body, strlen(body));
    if (n == 0) {
        return strdup(
            "assumptions: no list items detected. Use '- <item>' or "
            "'1. <item>' lines, one per assumption, then call again. "
            "Nothing was recorded.");
    }

    char *msg = malloc(640);
    if (!msg) return strdup("assumptions: out of memory");

    if (n >= 3 && plan_phase == PHASE_DRAFTING) {
        if (spike_cycles >= SPIKE_MAX_CYCLES) {
            snprintf(msg, 640,
                "assumptions: %d unverified item(s) — but already ran %d spike "
                "cycle(s) for this plan. No further spikes allowed; either (a) "
                "draft with what you know via plan_write, or (b) call "
                "spike_write with Decision: ABANDON.",
                n, SPIKE_MAX_CYCLES);
        } else {
            plan_phase = PHASE_SPIKE;
            spike_cycles++;
            spike_calls = 0;
            snprintf(msg, 640,
                "assumptions: %d unverified item(s) — routing to spike phase "
                "(cycle %d/%d). Use docs_*, code_context, read/grep/wc, "
                "web_search, web_fetch, readfile to investigate. Budget: %d tool calls / "
                "~%d tokens. When done, call spike_write with ## Question / "
                "## Findings / ## Decision: PROCEED-TO-PLAN | "
                "NEED-ANOTHER-SPIKE | ABANDON.",
                n, spike_cycles, SPIKE_MAX_CYCLES,
                SPIKE_MAX_CALLS, SPIKE_TOKEN_HINT);
        }
    } else if (n >= 3) {
        snprintf(msg, 640,
            "assumptions: %d unverified item(s) noted. Spike routing only "
            "triggers from drafting phase (current: %s).",
            n, plan_phase_name(plan_phase));
    } else {
        snprintf(msg, 640,
            "assumptions: %d unverified item(s) — under the spike threshold "
            "(3). Proceed to plan_write when ready.",
            n);
    }
    return msg;
}

/* Returns the trimmed Decision line value, or NULL. Caller frees. */
static char *spike_extract_decision(const char *body, size_t body_len) {
    const char *needle = "## Decision";
    size_t nlen = strlen(needle);
    const char *p = body, *end = body + body_len;
    while (p < end) {
        const char *le = p;
        while (le < end && *le != '\n') le++;
        if ((size_t)(le - p) >= nlen && strncmp(p, needle, nlen) == 0) {
            const char *q = (le < end) ? le + 1 : end;
            while (q < end) {
                const char *qe = q;
                while (qe < end && *qe != '\n') qe++;
                const char *t = q;
                while (t < qe && (*t == ' ' || *t == '\t')) t++;
                if (t < qe) {
                    const char *ee = qe;
                    while (ee > t && (ee[-1] == ' ' || ee[-1] == '\t' ||
                                       ee[-1] == '\r')) ee--;
                    char *out = malloc((size_t)(ee - t) + 1);
                    if (!out) return NULL;
                    memcpy(out, t, (size_t)(ee - t));
                    out[ee - t] = '\0';
                    return out;
                }
                q = (qe < end) ? qe + 1 : end;
            }
            return NULL;
        }
        p = (le < end) ? le + 1 : end;
    }
    return NULL;
}

char *validate_spike_artifact(const char *content, size_t content_len) {
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

    for (const char **k = SPIKE_REQUIRED_FRONTMATTER_KEYS; *k; k++) {
        if (!kb_fm_get(&fm, *k)) {
            sb_append_str(&errs, "- frontmatter missing required key: ");
            sb_append_str(&errs, *k);
            sb_append_char(&errs, '\n');
        }
    }
    const char *phase = kb_fm_get(&fm, "phase");
    if (phase && strcmp(phase, "spike") != 0) {
        sb_append_str(&errs, "- frontmatter 'phase' must be 'spike'\n");
    }
    const char *slug = kb_fm_get(&fm, "slug");
    if (slug && !kb_slug_valid(slug)) {
        sb_append_str(&errs, "- slug '");
        sb_append_str(&errs, slug);
        sb_append_str(&errs,
            "' invalid (lowercase a-z, 0-9, single hyphens; max 64 chars)\n");
    }

    const char *body = content + fm.body_offset;
    size_t body_len = content_len - fm.body_offset;
    for (const char **sec = SPIKE_SECTIONS; *sec; sec++) {
        if (!plan_has_h2(body, body_len, *sec)) {
            sb_append_str(&errs, "- missing required section: ## ");
            sb_append_str(&errs, *sec);
            sb_append_char(&errs, '\n');
        }
    }

    char *decision = spike_extract_decision(body, body_len);
    if (!decision) {
        sb_append_str(&errs,
            "- ## Decision section is empty; first non-blank line under it must be one of: "
            "PROCEED-TO-PLAN | NEED-ANOTHER-SPIKE | ABANDON\n");
    } else {
        bool ok = false;
        for (const char **d = SPIKE_VALID_DECISIONS; *d; d++) {
            if (strcmp(*d, decision) == 0) { ok = true; break; }
        }
        if (!ok) {
            sb_append_str(&errs, "- decision '");
            sb_append_str(&errs, decision);
            sb_append_str(&errs,
                "' invalid; must be one of: PROCEED-TO-PLAN | NEED-ANOTHER-SPIKE | ABANDON\n");
        }
        free(decision);
    }

    size_t lines = 1;
    for (size_t i = 0; i < content_len; i++) if (content[i] == '\n') lines++;
    if (lines > SPIKE_MAX_LINES) {
        char buf[128];
        snprintf(buf, sizeof(buf),
            "- artifact is %zu lines (max %d) — compress, don't truncate\n",
            lines, SPIKE_MAX_LINES);
        sb_append_str(&errs, buf);
    }

    kb_fm_free(&fm);
    if (errs.len == 0) {
        sb_free(&errs);
        return NULL;
    }
    return sb_to_str(&errs);
}

char *execute_spike_write(const char *body) {
    while (*body == ' ' || *body == '\t' || *body == '\n') body++;
    if (!*body) {
        return strdup(
            "spike_write not allowed: missing artifact body. See banner for "
            "the Question/Findings/Decision template.");
    }
    size_t body_len = strlen(body);

    char *errs = validate_spike_artifact(body, body_len);
    if (errs) {
        StringBuf out;
        sb_init(&out);
        sb_append_str(&out, "spike_write rejected — fix these and resubmit:\n");
        sb_append_str(&out, errs);
        free(errs);
        return sb_to_str(&out);
    }

    KbFrontmatter fm;
    kb_parse_frontmatter(body, body_len, &fm);
    const char *slug = kb_fm_get(&fm, "slug");
    if (!slug) {
        kb_fm_free(&fm);
        return strdup("spike_write internal: validator passed but slug missing");
    }
    if (current_plan_slug && strcmp(slug, current_plan_slug) != 0) {
        char *msg = malloc(384);
        snprintf(msg, 384,
            "spike_write rejected: slug '%s' does not match current plan '%s'.\n",
            slug, current_plan_slug);
        kb_fm_free(&fm);
        return msg;
    }

    if (kb_ensure_dirs() != 0) {
        kb_fm_free(&fm);
        char *msg = malloc(256);
        snprintf(msg, 256, "spike_write failed: cannot create %s (%s)\n",
                 KB_PLANS_DIR, strerror(errno));
        return msg;
    }

    char dest[1024];
    if ((size_t)snprintf(dest, sizeof(dest),
            "%s/%s.spike.md", KB_PLANS_DIR, slug) >= sizeof(dest)) {
        kb_fm_free(&fm);
        return strdup("spike_write failed: destination path too long");
    }

    /* Append on subsequent cycles, write fresh on the first. */
    struct stat st;
    bool exists = (stat(dest, &st) == 0);
    FILE *f = fopen(dest, exists ? "a" : "w");
    if (!f) {
        char *msg = malloc(1280);
        snprintf(msg, 1280, "spike_write failed: cannot open %s (%s)\n",
                 dest, strerror(errno));
        kb_fm_free(&fm);
        return msg;
    }
    if (exists) {
        fputs("\n---\n\n", f);
        /* On re-spike we keep only the body — the original frontmatter still
         * heads the file. */
        const char *append_body = body + fm.body_offset;
        size_t append_len = body_len - fm.body_offset;
        fwrite(append_body, 1, append_len, f);
        if (append_len == 0 || append_body[append_len - 1] != '\n') fputc('\n', f);
    } else {
        fwrite(body, 1, body_len, f);
        if (body_len == 0 || body[body_len - 1] != '\n') fputc('\n', f);
    }
    fclose(f);
    kb_fm_free(&fm);

    char *decision = spike_extract_decision(body, body_len);
    if (!decision) {
        return strdup(
            "spike_write internal: validator passed but Decision unreadable");
    }

    char *msg = malloc(1536);
    if (strcmp(decision, "PROCEED-TO-PLAN") == 0) {
        plan_phase = PHASE_DRAFTING;
        snprintf(msg, 1536,
            "Spike artifact written to %s. Decision: PROCEED-TO-PLAN — phase "
            "is now drafting. Call plan_write with the Proposal-A3 plan body.",
            dest);
    } else if (strcmp(decision, "NEED-ANOTHER-SPIKE") == 0) {
        if (spike_cycles >= SPIKE_MAX_CYCLES) {
            plan_phase = PHASE_NONE;
            free(current_plan_slug);
            current_plan_slug = NULL;
            spike_cycles = 0;
            spike_calls = 0;
            snprintf(msg, 1536,
                "Spike artifact written to %s. Decision: NEED-ANOTHER-SPIKE — "
                "but the %d-cycle cap is exhausted; phase forced to NONE. "
                "Re-enter with /plan <slug> if you have a new approach.",
                dest, SPIKE_MAX_CYCLES);
        } else {
            spike_cycles++;
            spike_calls = 0;
            snprintf(msg, 1536,
                "Spike artifact written to %s. Decision: NEED-ANOTHER-SPIKE — "
                "starting cycle %d/%d. Continue investigating; call "
                "spike_write again when you reach a final Decision.",
                dest, spike_cycles, SPIKE_MAX_CYCLES);
        }
    } else {  /* ABANDON */
        plan_phase = PHASE_NONE;
        free(current_plan_slug);
        current_plan_slug = NULL;
        spike_cycles = 0;
        spike_calls = 0;
        snprintf(msg, 1536,
            "Spike artifact written to %s. Decision: ABANDON — phase is now "
            "NONE; the plan slug is cleared. Tell the user why you abandoned.",
            dest);
    }
    free(decision);
    return msg;
}

/* ── Per-phase tool gate (Decision #5) ────────────────────────────── */

static size_t plan_tool_name(const char *command, char *out, size_t out_sz) {
    while (*command == ' ' || *command == '\t' || *command == '\n') command++;
    size_t i = 0;
    while (command[i] && command[i] != ' ' && command[i] != '\t' &&
           command[i] != '\n' && i + 1 < out_sz) {
        out[i] = command[i];
        i++;
    }
    out[i] = '\0';
    return i;
}

static bool plan_is_scaffold_list(const char *command) {
    while (*command == ' ' || *command == '\t' || *command == '\n') command++;
    if (strncmp(command, "scaffold", 8) != 0) return false;
    const char *a = command + 8;
    while (*a == ' ' || *a == '\t') a++;
    if (strncmp(a, "list", 4) != 0) return false;
    char b = a[4];
    return b == '\0' || b == ' ' || b == '\t' || b == '\n';
}

bool plan_tool_allowed(PlanPhase phase, const char *command) {
    char tool[32];
    plan_tool_name(command, tool, sizeof(tool));

    bool is_bash       = strcmp(tool, "bash") == 0;
    bool is_patch      = strcmp(tool, "edit") == 0;
    bool is_scaffold   = strcmp(tool, "scaffold") == 0;
    bool is_planwrite  = strcmp(tool, "plan_write") == 0;
    bool is_web        = strcmp(tool, "web_search") == 0 || strcmp(tool, "web_fetch") == 0;
    bool is_spikewrite = strcmp(tool, "spike_write") == 0;
    bool is_assume     = strcmp(tool, "assumptions") == 0;
    bool is_planverify = strcmp(tool, "plan_verify") == 0;

    if (is_scaffold && plan_is_scaffold_list(command)) return true;

    /* Spike-call budget: outside of spike_write itself, refuse further work
     * after SPIKE_MAX_CALLS so the model is forced to wrap up the spike. */
    if (phase == PHASE_SPIKE && !is_spikewrite &&
        spike_calls >= SPIKE_MAX_CALLS) {
        return false;
    }

    switch (phase) {
        case PHASE_NONE:
            return !(is_planwrite || is_spikewrite || is_assume || is_planverify);
        case PHASE_DRAFTING:
            return !(is_bash || is_patch || is_scaffold || is_spikewrite || is_planverify);
        case PHASE_SPIKE:
            return !(is_bash || is_patch || is_scaffold || is_planwrite ||
                     is_assume || is_planverify);
        case PHASE_PREMORTEM:
            return !(is_bash || is_patch || is_scaffold || is_web ||
                     is_spikewrite || is_assume || is_planverify);
        case PHASE_ACTIVE:
            return !(is_planwrite || is_spikewrite || is_assume);
    }
    return true;
}

char *plan_block_msg(PlanPhase phase, const char *command) {
    char tool[32];
    plan_tool_name(command, tool, sizeof(tool));

    StringBuf sb;
    sb_init(&sb);
    sb_append_str(&sb, tool[0] ? tool : "(empty)");
    sb_append_str(&sb, " not allowed: ");

    bool is_planwrite  = strcmp(tool, "plan_write") == 0;
    bool is_scaffold   = strcmp(tool, "scaffold") == 0;
    bool is_web        = strcmp(tool, "web_search") == 0 || strcmp(tool, "web_fetch") == 0;
    bool is_spikewrite = strcmp(tool, "spike_write") == 0;
    bool is_assume     = strcmp(tool, "assumptions") == 0;
    bool is_planverify = strcmp(tool, "plan_verify") == 0;

    /* Spike-call-budget exhaustion takes priority over the per-phase rules. */
    if (phase == PHASE_SPIKE && !is_spikewrite &&
        spike_calls >= SPIKE_MAX_CALLS) {
        char buf[160];
        snprintf(buf, sizeof(buf),
            "spike call budget exhausted (%d/%d). Call spike_write now with "
            "Decision: PROCEED-TO-PLAN, NEED-ANOTHER-SPIKE, or ABANDON.",
            spike_calls, SPIKE_MAX_CALLS);
        sb_append_str(&sb, buf);
        return sb_to_str(&sb);
    }

    if (is_planwrite && phase == PHASE_NONE) {
        sb_append_str(&sb,
            "only callable during drafting or premortem phase. Enter with /plan <slug>.");
    } else if (is_planwrite && phase == PHASE_SPIKE) {
        sb_append_str(&sb,
            "blocked in spike phase — finish the spike first via spike_write "
            "(Decision: PROCEED-TO-PLAN), then drafting resumes.");
    } else if (is_planwrite && phase == PHASE_ACTIVE) {
        sb_append_str(&sb,
            "plan already accepted; status updates not yet implemented.");
    } else if (is_spikewrite) {
        sb_append_str(&sb,
            "only callable during spike phase. Trigger one by listing ≥3 "
            "unverified items via the assumptions tool while drafting.");
    } else if (is_assume && phase == PHASE_SPIKE) {
        sb_append_str(&sb,
            "already in spike phase; you can't re-trigger from here. Finish "
            "the spike with spike_write.");
    } else if (is_assume) {
        sb_append_str(&sb,
            "only callable during drafting phase. Enter with /plan <slug>.");
    } else if (is_planverify) {
        sb_append_str(&sb,
            "only callable during active phase. Use /plan accept once the plan is approved.");
    } else if (is_scaffold) {
        sb_append_str(&sb, "blocked in ");
        sb_append_str(&sb, plan_phase_name(phase));
        sb_append_str(&sb,
            " phase (scaffold list is allowed). Use /plan accept to enter active phase, or /plan off to exit.");
    } else if (is_web && phase == PHASE_PREMORTEM) {
        sb_append_str(&sb,
            "blocked in premortem phase — review the existing plan, don't fetch new sources.");
    } else {
        sb_append_str(&sb, "blocked in ");
        sb_append_str(&sb, plan_phase_name(phase));
        sb_append_str(&sb,
            " phase. Use /plan accept to enter active phase, or /plan off to exit.");
    }
    return sb_to_str(&sb);
}
