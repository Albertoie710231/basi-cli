#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>

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

    bool is_bash      = strcmp(tool, "bash") == 0;
    bool is_patch     = strcmp(tool, "apply_patch") == 0;
    bool is_scaffold  = strcmp(tool, "scaffold") == 0;
    bool is_planwrite = strcmp(tool, "plan_write") == 0;
    bool is_webfetch  = strcmp(tool, "webfetch") == 0;

    if (is_scaffold && plan_is_scaffold_list(command)) return true;

    switch (phase) {
        case PHASE_NONE:
            return !is_planwrite;
        case PHASE_DRAFTING:
            return !(is_bash || is_patch || is_scaffold);
        case PHASE_SPIKE:
            return !(is_bash || is_patch || is_scaffold || is_planwrite);
        case PHASE_PREMORTEM:
            return !(is_bash || is_patch || is_scaffold || is_webfetch);
        case PHASE_ACTIVE:
            return !is_planwrite;
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

    bool is_planwrite = strcmp(tool, "plan_write") == 0;
    bool is_scaffold  = strcmp(tool, "scaffold") == 0;
    bool is_webfetch  = strcmp(tool, "webfetch") == 0;

    if (is_planwrite && phase == PHASE_NONE) {
        sb_append_str(&sb,
            "only callable during drafting or premortem phase. Enter with /plan <slug>.");
    } else if (is_planwrite && phase == PHASE_SPIKE) {
        sb_append_str(&sb,
            "blocked in spike phase — finish the spike first, then drafting begins.");
    } else if (is_planwrite && phase == PHASE_ACTIVE) {
        sb_append_str(&sb,
            "plan already accepted; status updates not yet implemented.");
    } else if (is_scaffold) {
        sb_append_str(&sb, "blocked in ");
        sb_append_str(&sb, plan_phase_name(phase));
        sb_append_str(&sb,
            " phase (scaffold list is allowed). Use /plan accept to enter active phase, or /plan off to exit.");
    } else if (is_webfetch && phase == PHASE_PREMORTEM) {
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
