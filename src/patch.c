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

/* ── apply_patch: Codex-style freeform diff ───────────────────────── */

typedef enum { OP_ADD, OP_DELETE, OP_UPDATE } PatchOpKind;

typedef struct {
    char *before;   /* concatenated " "+"-" lines (with newlines); empty for OP_ADD */
    char *after;    /* concatenated " "+"+" lines */
} PatchHunk;

typedef struct {
    PatchOpKind kind;
    char       *path;
    PatchHunk  *hunks;
    int         n_hunks;
} PatchFileOp;

typedef struct {
    PatchFileOp *ops;
    int          n_ops;
    char        *error;   /* NULL on success */
} ParsedPatch;

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

static void free_parsed_patch(ParsedPatch *p) {
    if (!p) return;
    for (int i = 0; i < p->n_ops; i++) {
        free(p->ops[i].path);
        for (int j = 0; j < p->ops[i].n_hunks; j++) {
            free(p->ops[i].hunks[j].before);
            free(p->ops[i].hunks[j].after);
        }
        free(p->ops[i].hunks);
    }
    free(p->ops);
    free(p->error);
    free(p);
}

static void flush_hunk(PatchFileOp *op, StringBuf *before, StringBuf *after) {
    op->hunks = realloc(op->hunks, sizeof(PatchHunk) * (op->n_hunks + 1));
    op->hunks[op->n_hunks].before = strdup(before->len ? before->data : "");
    op->hunks[op->n_hunks].after  = strdup(after->len  ? after->data  : "");
    op->n_hunks++;
    sb_clear(before);
    sb_clear(after);
}

static ParsedPatch *parse_patch(const char *text) {
    ParsedPatch *p = calloc(1, sizeof(*p));
    int op_cap = 0;
    PatchFileOp *cur = NULL;
    StringBuf before, after;
    sb_init(&before);
    sb_init(&after);
    bool in_hunk = false;

    while (*text == ' ' || *text == '\t' || *text == '\n') text++;

    char *line = take_line(&text);
    if (!line || strcmp(line, "*** Begin Patch") != 0) {
        p->error = strdup("apply_patch: missing '*** Begin Patch' marker on first line");
        free(line);
        sb_free(&before); sb_free(&after);
        return p;
    }
    free(line);

    while ((line = take_line(&text)) != NULL) {
        if (strcmp(line, "*** End Patch") == 0) {
            free(line);
            if (in_hunk && cur) flush_hunk(cur, &before, &after);
            sb_free(&before); sb_free(&after);
            return p;
        }

        if (strncmp(line, "*** ", 4) == 0) {
            if (in_hunk && cur) { flush_hunk(cur, &before, &after); in_hunk = false; }

            if (p->n_ops >= op_cap) {
                op_cap = op_cap ? op_cap * 2 : 4;
                p->ops = realloc(p->ops, sizeof(PatchFileOp) * op_cap);
            }
            cur = &p->ops[p->n_ops++];
            memset(cur, 0, sizeof(*cur));

            if (strncmp(line, "*** Add File: ", 14) == 0) {
                cur->kind = OP_ADD;
                cur->path = strdup(line + 14);
                in_hunk = true;
            } else if (strncmp(line, "*** Delete File: ", 17) == 0) {
                cur->kind = OP_DELETE;
                cur->path = strdup(line + 17);
                in_hunk = false;
            } else if (strncmp(line, "*** Update File: ", 17) == 0) {
                cur->kind = OP_UPDATE;
                cur->path = strdup(line + 17);
                in_hunk = false;
            } else {
                p->error = malloc(256);
                snprintf(p->error, 256, "apply_patch: unknown directive: %.180s", line);
                free(line);
                sb_free(&before); sb_free(&after);
                return p;
            }
            free(line);
            continue;
        }

        if (!cur) {
            p->error = strdup("apply_patch: content before any '*** Add/Delete/Update File:' directive");
            free(line);
            sb_free(&before); sb_free(&after);
            return p;
        }

        if (cur->kind == OP_DELETE) {
            if (line[0] != '\0') {
                p->error = strdup("apply_patch: 'Delete File' takes no body lines");
                free(line);
                sb_free(&before); sb_free(&after);
                return p;
            }
            free(line);
            continue;
        }

        if (strncmp(line, "@@", 2) == 0 && cur->kind == OP_UPDATE) {
            if (in_hunk) flush_hunk(cur, &before, &after);
            in_hunk = true;
            free(line);
            continue;
        }

        char prefix = line[0];
        const char *body = (prefix == '\0') ? "" : line + 1;

        if (cur->kind == OP_ADD) {
            if (prefix != '+' && prefix != '\0') {
                p->error = malloc(256);
                snprintf(p->error, 256,
                    "apply_patch: 'Add File' line must start with '+': %.180s", line);
                free(line);
                sb_free(&before); sb_free(&after);
                return p;
            }
            sb_append_str(&after, prefix == '\0' ? "" : body);
            sb_append_char(&after, '\n');
        } else { /* OP_UPDATE */
            in_hunk = true;
            if (prefix == ' ' || prefix == '\0') {
                sb_append_str(&before, prefix == '\0' ? "" : body);
                sb_append_char(&before, '\n');
                sb_append_str(&after,  prefix == '\0' ? "" : body);
                sb_append_char(&after,  '\n');
            } else if (prefix == '-') {
                sb_append_str(&before, body);
                sb_append_char(&before, '\n');
            } else if (prefix == '+') {
                sb_append_str(&after, body);
                sb_append_char(&after, '\n');
            } else {
                p->error = malloc(256);
                snprintf(p->error, 256,
                    "apply_patch: hunk line must start with ' ', '-', or '+': %.180s", line);
                free(line);
                sb_free(&before); sb_free(&after);
                return p;
            }
        }
        free(line);
    }

    /* EOF without explicit '*** End Patch' — accept and flush. */
    if (in_hunk && cur) flush_hunk(cur, &before, &after);
    sb_free(&before); sb_free(&after);
    return p;
}

static char *apply_add_file(const char *path, const char *content) {
    struct stat st;
    if (stat(path, &st) == 0) {
        char *e = malloc(512);
        snprintf(e, 512,
            "apply_patch: '%s' already exists. Use 'Update File' instead of 'Add File'.", path);
        return e;
    }
    char *path_copy = strdup(path);
    char *dir = dirname(path_copy);
    if (strcmp(dir, ".") != 0 && strcmp(dir, "/") != 0) mkdir_p(dir);
    free(path_copy);

    FILE *f = fopen(path, "w");
    if (!f) {
        char *e = malloc(512);
        snprintf(e, 512, "apply_patch: cannot create '%s' (%s)", path, strerror(errno));
        return e;
    }
    if (content && *content) fwrite(content, 1, strlen(content), f);
    fclose(f);
    return NULL;
}

static char *apply_delete_file(const char *path) {
    if (unlink(path) != 0) {
        char *e = malloc(512);
        snprintf(e, 512, "apply_patch: cannot delete '%s' (%s)", path, strerror(errno));
        return e;
    }
    return NULL;
}

static char *apply_update_file(const char *path, PatchFileOp *op) {
    FILE *f = fopen(path, "r");
    if (!f) {
        char *e = malloc(512);
        snprintf(e, 512, "apply_patch: cannot open '%s' for update (%s)", path, strerror(errno));
        return e;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *orig = malloc(fsize + 1);
    size_t nread = fread(orig, 1, fsize, f);
    orig[nread] = '\0';
    fclose(f);

    StringBuf cur;
    sb_init(&cur);
    sb_append(&cur, orig, nread);
    free(orig);

    for (int i = 0; i < op->n_hunks; i++) {
        const char *bef = op->hunks[i].before;
        const char *aft = op->hunks[i].after;

        if (!bef || !*bef) {
            char *e = malloc(512);
            snprintf(e, 512,
                "apply_patch: hunk %d in '%s' has no context — include at least one ' ' or '-' line.",
                i + 1, path);
            sb_free(&cur);
            return e;
        }

        char *match = memmem(cur.data, cur.len, bef, strlen(bef));
        if (!match) {
            char *e = malloc(1024);
            snprintf(e, 1024,
                "apply_patch: hunk %d in '%s' — context not found. Re-read the file and verify the ' ' and '-' lines match exactly (including whitespace).",
                i + 1, path);
            sb_free(&cur);
            return e;
        }
        size_t bef_len = strlen(bef);
        char *next = memmem(match + 1, cur.len - (match - cur.data) - 1, bef, bef_len);
        if (next) {
            char *e = malloc(1024);
            snprintf(e, 1024,
                "apply_patch: hunk %d in '%s' — context matches multiple locations. Add more surrounding ' ' lines to disambiguate.",
                i + 1, path);
            sb_free(&cur);
            return e;
        }

        size_t aft_len = strlen(aft);
        size_t prefix_len = match - cur.data;
        size_t suffix_len = cur.len - prefix_len - bef_len;

        StringBuf nxt;
        sb_init(&nxt);
        sb_append(&nxt, cur.data, prefix_len);
        sb_append(&nxt, aft, aft_len);
        sb_append(&nxt, cur.data + prefix_len + bef_len, suffix_len);
        sb_free(&cur);
        cur = nxt;
    }

    f = fopen(path, "w");
    if (!f) {
        char *e = malloc(512);
        snprintf(e, 512, "apply_patch: cannot write '%s' (%s)", path, strerror(errno));
        sb_free(&cur);
        return e;
    }
    if (cur.len) fwrite(cur.data, 1, cur.len, f);
    fclose(f);
    sb_free(&cur);
    return NULL;
}

char *execute_apply_patch(const char *patch_text) {
    ParsedPatch *p = parse_patch(patch_text);
    if (p->error) {
        char *err = strdup(p->error);
        free_parsed_patch(p);
        return err;
    }
    if (p->n_ops == 0) {
        free_parsed_patch(p);
        return strdup("apply_patch: empty patch (no Add/Delete/Update File directives)");
    }

    StringBuf summary;
    sb_init(&summary);
    for (int i = 0; i < p->n_ops; i++) {
        char line[512];
        const char *kind = p->ops[i].kind == OP_ADD    ? "add"
                         : p->ops[i].kind == OP_DELETE ? "delete"
                         :                               "edit";
        if (p->ops[i].kind == OP_UPDATE) {
            snprintf(line, sizeof(line), "%s%s %s (%d hunk%s)",
                i ? ", " : "", kind, p->ops[i].path, p->ops[i].n_hunks,
                p->ops[i].n_hunks == 1 ? "" : "s");
        } else {
            snprintf(line, sizeof(line), "%s%s %s",
                i ? ", " : "", kind, p->ops[i].path);
        }
        sb_append_str(&summary, line);
    }

    bool ap_auto = (permission_mode == PERM_BYPASS)
                || (permission_mode == PERM_ACCEPT_EDITS)
                || apply_patch_always_allowed;
    if (!ap_auto) {
        printf("\n\033[36m%s\033[0m\n", patch_text);
        char *summary_str = sb_to_str(&summary);
        int decision = request_approval("apply_patch", summary_str);
        free(summary_str);
        if (decision == 0) {
            free_parsed_patch(p);
            return strdup("User denied execution.");
        }
        if (decision == 2) apply_patch_always_allowed = true;
    } else {
        sb_free(&summary);
    }

    StringBuf result;
    sb_init(&result);
    int succeeded = 0;
    for (int i = 0; i < p->n_ops; i++) {
        char *err = NULL;
        if (p->ops[i].kind == OP_ADD) {
            const char *content = (p->ops[i].n_hunks > 0) ? p->ops[i].hunks[0].after : "";
            err = apply_add_file(p->ops[i].path, content);
        } else if (p->ops[i].kind == OP_DELETE) {
            err = apply_delete_file(p->ops[i].path);
        } else {
            err = apply_update_file(p->ops[i].path, &p->ops[i]);
        }

        if (err) {
            sb_append_str(&result, err);
            sb_append_char(&result, '\n');
            free(err);
            if (succeeded) sb_append_str(&result, "WARNING: patch was partially applied; earlier files were modified.\n");
            free_parsed_patch(p);
            return sb_to_str(&result);
        }
        char ok[512];
        const char *kind = p->ops[i].kind == OP_ADD    ? "Added"
                         : p->ops[i].kind == OP_DELETE ? "Deleted"
                         :                               "Updated";
        snprintf(ok, sizeof(ok), "%s %s\n", kind, p->ops[i].path);
        sb_append_str(&result, ok);
        succeeded++;
    }

    free_parsed_patch(p);
    return sb_to_str(&result);
}
