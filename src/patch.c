#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <libgen.h>
#include <sys/stat.h>
#include <errno.h>

#include "util.h"
#include "globals.h"
#include "patch.h"
#include "reuse.h"

/* ── edit: SEARCH/REPLACE block tool ──────────────────────────────────
 * The model expresses a change as old-text -> new-text, never a diff. We
 * locate the old text deterministically with a cascade:
 *   1. exact substring match (handles whole blocks AND fragments within a
 *      line — "hello" matches inside "hello world")
 *   2. line-trimmed match (whitespace-insensitive, for multi-line blocks
 *      whose indentation/trailing space drifted)
 * On a fuzzy match, added lines are re-indented to the file's real
 * indentation; nothing in the surrounding file is rewritten. We deliberately
 * do NOT do block-anchor/Levenshtein guessing — that's the strategy that can
 * swallow a large span, and it would break the "never corrupt" contract.
 *
 * Grammar (inside one <tool>edit ...</tool>):
 *   edit <path>
 *   <<<<<<< SEARCH
 *   <text to find>
 *   =======
 *   <replacement text>
 *   >>>>>>> REPLACE
 * Multiple blocks may follow, applied in order. An empty SEARCH block writes the
 * whole file from the REPLACE text — creating it if new, or overwriting it if it
 * already exists (read-before-edit still applies to existing files). */

/* A file must be read this session before it can be edited, unless it is
 * larger than the read tool would ever surface (~2000 tokens), in which case
 * it could not have been read and we must not trap the model. */
#define READ_GATE_MAX_BYTES (2000 * 4)

typedef struct {
    char *find;   /* old text, no trailing newline; "" means create-file */
    char *repl;   /* new text, no trailing newline */
} EditBlock;

typedef struct {
    char      *path;
    EditBlock *blocks;
    int        n_blocks;
    char      *error;     /* NULL on success */
} ParsedEdit;

static char *take_line(const char **p) {
    if (!**p) return NULL;
    const char *start = *p;
    const char *nl = strchr(start, '\n');
    size_t len = nl ? (size_t)(nl - start) : strlen(start);
    char *line = malloc(len + 1);
    memcpy(line, start, len);
    line[len] = '\0';
    *p = nl ? nl + 1 : start + len;
    return line;
}

static char *trim_dup(const char *s) {
    while (*s == ' ' || *s == '\t') s++;
    const char *e = s + strlen(s);
    while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r')) e--;
    char *out = malloc((size_t)(e - s) + 1);
    memcpy(out, s, (size_t)(e - s));
    out[e - s] = '\0';
    return out;
}

static void free_parsed_edit(ParsedEdit *p) {
    if (!p) return;
    free(p->path);
    for (int i = 0; i < p->n_blocks; i++) {
        free(p->blocks[i].find);
        free(p->blocks[i].repl);
    }
    free(p->blocks);
    free(p->error);
    free(p);
}

static bool is_marker(const char *line, char c) {
    while (*line == ' ' || *line == '\t') line++;
    for (int i = 0; i < 7; i++) if (line[i] != c) return false;
    return true;
}

static void push_block(ParsedEdit *p, int *cap, StringBuf *find_sb, StringBuf *repl_sb) {
    if (p->n_blocks >= *cap) {
        *cap = *cap ? *cap * 2 : 4;
        p->blocks = realloc(p->blocks, sizeof(EditBlock) * (*cap));
    }
    p->blocks[p->n_blocks].find = sb_to_str(find_sb);
    p->blocks[p->n_blocks].repl = sb_to_str(repl_sb);
    p->n_blocks++;
    sb_init(find_sb); sb_init(repl_sb);   /* data now owned by the block */
}

static void repl_append(StringBuf *repl_sb, bool *repl_first, const char *s) {
    if (!*repl_first) sb_append_char(repl_sb, '\n');
    *repl_first = false;
    sb_append_str(repl_sb, s);
}

static ParsedEdit *parse_edit(const char *text) {
    ParsedEdit *p = calloc(1, sizeof(*p));

    while (*text == ' ' || *text == '\t' || *text == '\n') text++;
    char *path_line = take_line(&text);
    if (!path_line || !*path_line) {
        p->error = strdup("edit: missing file path. Format: edit <path> then a SEARCH/REPLACE block.");
        free(path_line);
        return p;
    }
    p->path = trim_dup(path_line);
    free(path_line);

    enum { OUT, IN_FIND, IN_REPL } st = OUT;
    StringBuf find_sb, repl_sb;
    sb_init(&find_sb);
    sb_init(&repl_sb);
    bool find_first = true, repl_first = true;
    /* A '=======' seen inside the REPLACE section is held here rather than
     * committed immediately: models routinely emit a stray extra divider just
     * before '>>>>>>> REPLACE', or use '=======' as the closer with no
     * '>>>>>>>' at all. We only fold it into the replacement if real content
     * follows it; otherwise it is discarded as a stray/closer marker. */
    char *pending_div = NULL;
    int cap = 0;
    char *line;

    while ((line = take_line(&text)) != NULL) {
        if (st == OUT && is_marker(line, '<')) {
            st = IN_FIND;
            sb_clear(&find_sb); sb_clear(&repl_sb);
            find_first = repl_first = true;
            free(pending_div); pending_div = NULL;
            free(line); continue;
        }
        if (st == IN_FIND && is_marker(line, '=')) {
            st = IN_REPL; free(line); continue;
        }
        if (st == IN_REPL && is_marker(line, '>')) {       /* hard close */
            free(pending_div); pending_div = NULL;          /* trailing divider was stray */
            push_block(p, &cap, &find_sb, &repl_sb);
            st = OUT; free(line); continue;
        }
        if (st == IN_REPL && is_marker(line, '=')) {        /* defer divider */
            if (pending_div) { repl_append(&repl_sb, &repl_first, pending_div); free(pending_div); }
            pending_div = strdup(line);
            free(line); continue;
        }
        if (st == IN_FIND) {
            if (!find_first) sb_append_char(&find_sb, '\n');
            find_first = false;
            sb_append_str(&find_sb, line);
        } else if (st == IN_REPL) {
            if (pending_div) {                              /* divider was real content */
                repl_append(&repl_sb, &repl_first, pending_div);
                free(pending_div); pending_div = NULL;
            }
            repl_append(&repl_sb, &repl_first, line);
        }
        /* lines outside a block are ignored */
        free(line);
    }

    /* EOF: a pending divider that was never followed by content acted as the
     * block's closer (the '======='-as-terminator pattern) — finalize. */
    if (st == IN_REPL && pending_div) {
        free(pending_div); pending_div = NULL;
        push_block(p, &cap, &find_sb, &repl_sb);
        st = OUT;
    }
    free(pending_div);

    if (st != OUT)
        p->error = strdup("edit: unterminated block — a SEARCH needs a '=======' divider then a '>>>>>>> REPLACE' line.");
    else if (p->n_blocks == 0)
        p->error = strdup("edit: no SEARCH/REPLACE block found. Use:\n<<<<<<< SEARCH\n<text to find>\n=======\n<replacement>\n>>>>>>> REPLACE");

    sb_free(&find_sb);
    sb_free(&repl_sb);
    return p;
}

/* ── whitespace-insensitive line helpers (fuzzy fallback) ──────────── */

static size_t lead_ws(const char *s) {
    size_t i = 0;
    while (s[i] == ' ' || s[i] == '\t') i++;
    return i;
}

static const char *trim_bounds(const char *s, size_t *out_len) {
    const char *a = s;
    while (*a == ' ' || *a == '\t' || *a == '\r') a++;
    const char *e = s + strlen(s);
    while (e > a && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r')) e--;
    *out_len = (size_t)(e - a);
    return a;
}

static bool trim_eq(const char *x, const char *y) {
    size_t lx, ly;
    const char *px = trim_bounds(x, &lx);
    const char *py = trim_bounds(y, &ly);
    return lx == ly && memcmp(px, py, lx) == 0;
}

typedef struct { size_t off; char *text; bool has_nl; } FileLine;

static FileLine *split_lines(const char *data, size_t len, int *out_n) {
    int cap = 64, n = 0;
    FileLine *arr = malloc(sizeof(FileLine) * cap);
    size_t i = 0;
    while (i < len) {
        size_t start = i;
        while (i < len && data[i] != '\n') i++;
        size_t linelen = i - start;
        bool has_nl = (i < len);
        if (n >= cap) { cap *= 2; arr = realloc(arr, sizeof(FileLine) * cap); }
        arr[n].off = start;
        arr[n].text = malloc(linelen + 1);
        memcpy(arr[n].text, data + start, linelen);
        arr[n].text[linelen] = '\0';
        arr[n].has_nl = has_nl;
        n++;
        if (has_nl) i++; else break;
    }
    *out_n = n;
    return arr;
}

static void free_lines(FileLine *arr, int n) {
    for (int i = 0; i < n; i++) free(arr[i].text);
    free(arr);
}

static char **split_str_lines(const char *s, int *out_n) {
    int cap = 8, n = 0;
    char **arr = malloc(sizeof(char *) * cap);
    const char *p = s;
    for (;;) {
        const char *nl = strchr(p, '\n');
        size_t len = nl ? (size_t)(nl - p) : strlen(p);
        if (n >= cap) { cap *= 2; arr = realloc(arr, sizeof(char *) * cap); }
        arr[n] = malloc(len + 1);
        memcpy(arr[n], p, len);
        arr[n][len] = '\0';
        n++;
        if (!nl) break;
        p = nl + 1;
    }
    *out_n = n;
    return arr;
}

/* Re-indent one replacement line so its leading whitespace matches the file's
 * real indentation at the match site (inferred from the first matched line). */
static char *reindent_line(const char *repl, const char *model_ref, const char *file_ref) {
    size_t mlen = lead_ws(model_ref);
    size_t flen = lead_ws(file_ref);
    if (mlen > 0 && strncmp(repl, model_ref, mlen) == 0) {
        StringBuf b; sb_init(&b);
        sb_append(&b, file_ref, flen);
        sb_append_str(&b, repl + mlen);
        return sb_to_str(&b);
    }
    if (mlen == 0 && flen > 0) {
        StringBuf b; sb_init(&b);
        sb_append(&b, file_ref, flen);
        sb_append_str(&b, repl);
        return sb_to_str(&b);
    }
    return strdup(repl);
}

/* Replace one occurrence of `find` with `repl` in `cur`. Returns NULL on
 * success, else a malloc'd model-facing error. */
static char *replace_one(StringBuf *cur, const char *find, const char *repl,
                         const char *path, int idx) {
    /* 1. exact substring (primary). */
    size_t find_len = strlen(find);
    char *match = memmem(cur->data, cur->len, find, find_len);
    if (match) {
        char *next = memmem(match + 1, cur->len - (match - cur->data) - 1, find, find_len);
        if (next) {
            char *e = malloc(768);
            snprintf(e, 768,
                "edit: block %d — the SEARCH text matches more than one place in '%s'. Add a few surrounding lines so it is unique.",
                idx + 1, path);
            return e;
        }
        size_t prefix_len = match - cur->data;
        size_t suffix_len = cur->len - prefix_len - find_len;
        StringBuf nxt; sb_init(&nxt);
        sb_append(&nxt, cur->data, prefix_len);
        sb_append_str(&nxt, repl);
        sb_append(&nxt, cur->data + prefix_len + find_len, suffix_len);
        sb_free(cur);
        *cur = nxt;
        return NULL;
    }

    /* 2. line-trimmed fallback (whole-line blocks with whitespace drift). */
    int n_find = 0;
    char **find_lines = split_str_lines(find, &n_find);
    int n_file = 0;
    FileLine *fl = split_lines(cur->data, cur->len, &n_file);

    int found_at = -1, n_found = 0;
    for (int s = 0; s + n_find <= n_file; s++) {
        bool ok = true;
        for (int k = 0; k < n_find; k++)
            if (!trim_eq(fl[s + k].text, find_lines[k])) { ok = false; break; }
        if (ok) { n_found++; if (found_at < 0) found_at = s; }
    }

    if (n_found != 1) {
        char *e = malloc(768);
        if (n_found == 0)
            snprintf(e, 768,
                "edit: block %d — the SEARCH text was not found in '%s'. Copy it verbatim from the file (use read/grep), or include more surrounding lines.",
                idx + 1, path);
        else
            snprintf(e, 768,
                "edit: block %d — the SEARCH text matches more than one place in '%s' (even ignoring whitespace). Add surrounding lines to make it unique.",
                idx + 1, path);
        for (int i = 0; i < n_find; i++) free(find_lines[i]);
        free(find_lines); free_lines(fl, n_file);
        return e;
    }

    int s = found_at, e_line = found_at + n_find;
    size_t start_off = fl[s].off;
    size_t end_off   = fl[e_line - 1].off + strlen(fl[e_line - 1].text)
                     + (fl[e_line - 1].has_nl ? 1 : 0);
    bool span_had_nl = fl[e_line - 1].has_nl;

    int n_repl = 0;
    char **repl_lines = split_str_lines(repl, &n_repl);
    StringBuf rep; sb_init(&rep);
    for (int i = 0; i < n_repl; i++) {
        char *ln = reindent_line(repl_lines[i], find_lines[0], fl[s].text);
        if (i) sb_append_char(&rep, '\n');
        sb_append_str(&rep, ln);
        free(ln);
    }
    if (span_had_nl) sb_append_char(&rep, '\n');

    StringBuf nxt; sb_init(&nxt);
    sb_append(&nxt, cur->data, start_off);
    sb_append(&nxt, rep.data ? rep.data : "", rep.len);
    sb_append(&nxt, cur->data + end_off, cur->len - end_off);
    sb_free(cur);
    *cur = nxt;

    sb_free(&rep);
    for (int i = 0; i < n_repl; i++) free(repl_lines[i]);
    free(repl_lines);
    for (int i = 0; i < n_find; i++) free(find_lines[i]);
    free(find_lines);
    free_lines(fl, n_file);
    return NULL;
}

char *execute_edit(const char *args) {
    ParsedEdit *p = parse_edit(args);
    if (p->error) {
        char *err = strdup(p->error);
        free_parsed_edit(p);
        return err;
    }

    const char *path = p->path;
    struct stat st;
    bool file_existed = (stat(path, &st) == 0);

    /* Read-before-edit invariant: an existing, readable file must have been
     * viewed this session. New files (create) and oversize files are exempt. */
    if (file_existed && !read_tracker_seen(path) && st.st_size <= READ_GATE_MAX_BYTES) {
        char *e = malloc(640);
        snprintf(e, 640,
            "edit: you have not read '%s' this session. Read it first (e.g. 'read %s', or head/grep for the relevant part) so the SEARCH text matches the real file, then re-apply.",
            path, path);
        free_parsed_edit(p);
        return e;
    }

    /* Reject no-op blocks early (identical find/replace). */
    for (int i = 0; i < p->n_blocks; i++) {
        if (p->blocks[i].find[0] && strcmp(p->blocks[i].find, p->blocks[i].repl) == 0) {
            char *e = malloc(256);
            snprintf(e, 256, "edit: block %d — SEARCH and REPLACE are identical; nothing to change.", i + 1);
            free_parsed_edit(p);
            return e;
        }
    }

    /* Reuse gate (opt-in via BASI_REUSE_GATE): before an edit ADDS a new
     * top-level function, pause if a near-duplicate already exists elsewhere in
     * the tree. Warn-once per function name — a re-issue applies. No-op when the
     * gate is off or no embedder is present, so editing never depends on it. */
    int autofixed = 0;
    if (reuse_gate_enabled()) {
        for (int i = 0; i < p->n_blocks; i++) {
            /* First try the deterministic rewrite (output-side injection); if it
             * fires, the reusing edit is applied instead of the duplicate. */
            char *fixed = reuse_gate_autofix(path, p->blocks[i].repl, p->blocks[i].find);
            if (fixed) { free(p->blocks[i].repl); p->blocks[i].repl = fixed; autofixed++; continue; }
            char *warn = reuse_gate_check(path, p->blocks[i].repl, p->blocks[i].find);
            if (warn) { free_parsed_edit(p); return warn; }
        }
    }

    /* Approval (same policy as before: bypass / accept-edits auto-approve). */
    bool ap_auto = (permission_mode == PERM_BYPASS)
                || (permission_mode == PERM_ACCEPT_EDITS)
                || apply_patch_always_allowed;
    if (!ap_auto) {
        printf("\n\033[36medit %s\033[0m\n%s\n", path, args);
        char summary[512];
        snprintf(summary, sizeof(summary), "%s %s (%d change%s)",
            file_existed ? "edit" : "create", path,
            p->n_blocks, p->n_blocks == 1 ? "" : "s");
        int decision = request_approval("edit", summary);
        if (decision == 0) { free_parsed_edit(p); return strdup("User denied execution."); }
        if (decision == 2) apply_patch_always_allowed = true;
    }

    /* Load current content (empty buffer for a not-yet-existing file). */
    StringBuf cur; sb_init(&cur);
    if (file_existed) {
        size_t nread = 0;
        char *orig = read_file_all(path, &nread);   /* shared, ftell-guarded slurp */
        if (!orig) {
            char *e = malloc(512);
            snprintf(e, 512, "edit: cannot open '%s' (%s)", path, strerror(errno));
            free_parsed_edit(p);
            return e;
        }
        sb_append(&cur, orig, nread);
        free(orig);
    }

    bool created = false;
    for (int i = 0; i < p->n_blocks; i++) {
        const char *find = p->blocks[i].find;
        const char *repl = p->blocks[i].repl;
        if (!find[0]) {
            /* Empty SEARCH = write the WHOLE file: create it if new, or overwrite
               it wholesale if it already exists. The read-before-edit gate above
               already required an existing file to have been viewed this session,
               so this is a deliberate "rewrite this file" (a write capability),
               not a blind clobber — it lets a model replace a stub without having
               to quote the stub's exact current contents. */
            sb_clear(&cur);
            sb_append_str(&cur, repl);
            if (cur.len && cur.data[cur.len - 1] != '\n') sb_append_char(&cur, '\n');
            created = true;
            continue;
        }
        if (cur.len == 0) {
            sb_free(&cur);
            char *e = malloc(384);
            snprintf(e, 384, "edit: block %d — '%s' is empty or does not exist; use an empty SEARCH block to create it.", i + 1, path);
            free_parsed_edit(p);
            return e;
        }
        char *err = replace_one(&cur, find, repl, path, i);
        if (err) { sb_free(&cur); free_parsed_edit(p); return err; }
    }

    /* Behavior-preservation guard (opt-in BASI_REUSE_REGRESS): before writing,
     * check whether this edit silently changed the behavior of a function that
     * already existed. The old content is still on disk (we write below), so we
     * re-read it and compare against the assembled new content. Warn-once per
     * function name — a re-issue applies. No-op when disabled or the file is new. */
    if (file_existed && reuse_regress_enabled()) {
        size_t olen = 0;
        char *oldc = read_file_all(path, &olen);   /* disk still holds the pre-edit content */
        if (oldc) {
            char *newc = malloc(cur.len + 1);
            memcpy(newc, cur.data, cur.len); newc[cur.len] = '\0';
            char *rwarn = reuse_regress_check(path, oldc, newc);
            free(oldc); free(newc);
            if (rwarn) { sb_free(&cur); free_parsed_edit(p); return rwarn; }
        }
    }

    if (!file_existed) {
        char *pc = strdup(path);
        char *dir = dirname(pc);
        if (strcmp(dir, ".") != 0 && strcmp(dir, "/") != 0) mkdir_p(dir);
        free(pc);
    }
    FILE *f = fopen(path, "w");
    if (!f) {
        char *e = malloc(512);
        snprintf(e, 512, "edit: cannot write '%s' (%s)", path, strerror(errno));
        sb_free(&cur);
        free_parsed_edit(p);
        return e;
    }
    if (cur.len) fwrite(cur.data, 1, cur.len, f);
    fclose(f);

    char *result = malloc(320);
    int rn = snprintf(result, 320, "%s %s (%d block%s applied)",
        (created && !file_existed) ? "Created" : "Edited",
        path, p->n_blocks, p->n_blocks == 1 ? "" : "s");
    if (autofixed > 0 && rn > 0 && rn < 320)
        snprintf(result + rn, 320 - rn,
            " [reuse-gate rewrote %d function%s to call existing code]",
            autofixed, autofixed == 1 ? "" : "s");
    sb_free(&cur);
    free_parsed_edit(p);
    return result;
}
