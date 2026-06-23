#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>

#include "util.h"
#include "kb.h"

/* ── Knowledge base / plans storage ───────────────────────────────────
 *
 * Filesystem layout (decision #2 in .basi/plans/knowledge-base.md):
 *
 *   .basi/
 *     plans/                       plan + spike artifacts
 *     knowledge/
 *       notes/                     user notes (HIGHEST precedence)
 *       pinned/                    explicitly added external sources
 *       docs/                      bulk-imported official docs (LOWEST)
 *
 * All knowledge files are markdown with YAML frontmatter. Every util in
 * this section is read-only-safe and idempotent — invoking before any
 * dir exists is fine.
 */



const char *kb_shelf_name(KbShelf s) {
    switch (s) {
        case KB_SHELF_NOTES:  return "notes";
        case KB_SHELF_PINNED: return "pinned";
        case KB_SHELF_DOCS:   return "docs";
        default:              return "unknown";
    }
}

const char *kb_shelf_dir(KbShelf s) {
    switch (s) {
        case KB_SHELF_NOTES:  return KB_NOTES_DIR;
        case KB_SHELF_PINNED: return KB_PINNED_DIR;
        case KB_SHELF_DOCS:   return KB_DOCS_DIR;
        default:              return NULL;
    }
}

KbShelf kb_shelf_from_name(const char *s) {
    if (!s) return KB_SHELF_UNKNOWN;
    if (strcmp(s, "notes") == 0)  return KB_SHELF_NOTES;
    if (strcmp(s, "pinned") == 0) return KB_SHELF_PINNED;
    if (strcmp(s, "docs") == 0)   return KB_SHELF_DOCS;
    return KB_SHELF_UNKNOWN;
}

/* Create all KB directories. Idempotent.
 * Returns 0 on success, -1 on failure (errno set). */
int kb_ensure_dirs(void) {
    if (mkdir_p(KB_ROOT)       != 0) return -1;
    if (mkdir_p(KB_PLANS_DIR)  != 0) return -1;
    if (mkdir_p(KB_KNOW_DIR)   != 0) return -1;
    if (mkdir_p(KB_NOTES_DIR)  != 0) return -1;
    if (mkdir_p(KB_PINNED_DIR) != 0) return -1;
    if (mkdir_p(KB_DOCS_DIR)   != 0) return -1;
    return 0;
}

/* Slug validator: lowercase letters + digits + single hyphens, 1-64 chars,
 * must start with a letter, no leading/trailing/double hyphens.
 * Used for plan filenames (./.basi/plans/<slug>.md). */
bool kb_slug_valid(const char *s) {
    if (!s || !*s) return false;
    size_t n = strlen(s);
    if (n > 64) return false;
    if (!(s[0] >= 'a' && s[0] <= 'z')) return false;
    if (s[n-1] == '-') return false;
    for (size_t i = 0; i < n; i++) {
        char c = s[i];
        bool ok = (c >= 'a' && c <= 'z') ||
                  (c >= '0' && c <= '9') ||
                  (c == '-');
        if (!ok) return false;
        if (c == '-' && i > 0 && s[i-1] == '-') return false;
    }
    return true;
}

/* Frontmatter parser. Only flat key:value pairs (no nesting, no lists).
 *
 * Format:
 *   ---\n
 *   key: value\n
 *   key2: value2\n
 *   ---\n
 *   <body>
 *
 * If the input doesn't start with "---\n", it has no frontmatter and the
 * parse returns 0 with an empty entry list and body_offset = 0.
 * If "---" opens but never closes within `len`, returns -1.
 */


void kb_fm_init(KbFrontmatter *fm) {
    fm->entries = NULL;
    fm->count = fm->cap = 0;
    fm->body_offset = 0;
}

void kb_fm_free(KbFrontmatter *fm) {
    for (size_t i = 0; i < fm->count; i++) {
        free(fm->entries[i].key);
        free(fm->entries[i].value);
    }
    free(fm->entries);
    fm->entries = NULL;
    fm->count = fm->cap = 0;
    fm->body_offset = 0;
}

static void kb_fm_push(KbFrontmatter *fm,
                       const char *k, size_t klen,
                       const char *v, size_t vlen) {
    if (fm->count == fm->cap) {
        fm->cap = fm->cap ? fm->cap * 2 : 8;
        fm->entries = realloc(fm->entries, fm->cap * sizeof(KbFmEntry));
    }
    KbFmEntry *e = &fm->entries[fm->count++];
    e->key = malloc(klen + 1);
    memcpy(e->key, k, klen);
    e->key[klen] = '\0';
    e->value = malloc(vlen + 1);
    memcpy(e->value, v, vlen);
    e->value[vlen] = '\0';
}

const char *kb_fm_get(const KbFrontmatter *fm, const char *key) {
    for (size_t i = 0; i < fm->count; i++) {
        if (strcmp(fm->entries[i].key, key) == 0) return fm->entries[i].value;
    }
    return NULL;
}

int kb_parse_frontmatter(const char *text, size_t len, KbFrontmatter *out) {
    kb_fm_init(out);
    if (!text || len < 4) return 0;
    if (text[0] != '-' || text[1] != '-' || text[2] != '-') return 0;

    size_t i = 3;
    if (i < len && text[i] == '\r') i++;
    if (i >= len || text[i] != '\n') return 0;
    i++;

    while (i < len) {
        /* Closing fence? */
        if (i + 3 <= len &&
            text[i] == '-' && text[i+1] == '-' && text[i+2] == '-') {
            size_t j = i + 3;
            if (j < len && text[j] == '\r') j++;
            if (j >= len || text[j] == '\n') {
                out->body_offset = (j < len) ? j + 1 : len;
                return 0;
            }
        }

        size_t eol = i;
        while (eol < len && text[eol] != '\n') eol++;

        size_t colon = i;
        while (colon < eol && text[colon] != ':') colon++;
        if (colon < eol) {
            size_t ks = i, ke = colon;
            while (ks < ke && (text[ks] == ' ' || text[ks] == '\t')) ks++;
            while (ke > ks && (text[ke-1] == ' ' || text[ke-1] == '\t' || text[ke-1] == '\r')) ke--;

            size_t vs = colon + 1, ve = eol;
            while (vs < ve && (text[vs] == ' ' || text[vs] == '\t')) vs++;
            while (ve > vs && (text[ve-1] == ' ' || text[ve-1] == '\t' || text[ve-1] == '\r')) ve--;

            if (ke > ks) kb_fm_push(out, text + ks, ke - ks, text + vs, ve - vs);
        }

        i = (eol < len) ? eol + 1 : eol;
    }

    /* Hit EOF before closing fence. */
    kb_fm_free(out);
    return -1;
}

/* ── Knowledge-base librarian toolkit (read-only) ──────────────────── */

/* Read a whole file into a malloc'd buffer; returns NULL on error.
 * *out_len is filled (excludes the appended '\0'). Caller frees.
 * Thin wrapper over the shared read_file_all() so there is one slurp. */
char *kb_read_file(const char *path, size_t *out_len) {
    return read_file_all(path, out_len);
}

/* Visit every .md file under `dir` recursively, calling `visit(path, ud)`.
 * Missing dir is silently OK (returns 0). Visitor returning non-zero stops
 * the walk early. */

int kb_walk_dir(const char *dir, KbVisitor visit, void *ud) {
    DIR *d = opendir(dir);
    if (!d) return 0;
    struct dirent *de;
    int rc = 0;
    while ((de = readdir(d)) != NULL) {
        if (de->d_name[0] == '.') continue;
        char path[1024];
        if ((size_t)snprintf(path, sizeof(path), "%s/%s", dir, de->d_name) >= sizeof(path)) continue;
        struct stat st;
        if (stat(path, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            rc = kb_walk_dir(path, visit, ud);
            if (rc) break;
        } else if (S_ISREG(st.st_mode)) {
            size_t n = strlen(de->d_name);
            if (n > 3 && strcmp(de->d_name + n - 3, ".md") == 0) {
                rc = visit(path, ud);
                if (rc) break;
            }
        }
    }
    closedir(d);
    return rc;
}

/* Strip ".basi/knowledge/" prefix from a path; returns the suffix or the
 * original pointer if no match. */
const char *kb_strip_know_prefix(const char *path) {
    size_t plen = strlen(KB_KNOW_DIR);
    if (strncmp(path, KB_KNOW_DIR "/", plen + 1) == 0) return path + plen + 1;
    return path;
}

/* ─── docs_toc ─────────────────────────────────────────────────────── */

typedef struct { StringBuf *out; size_t count; size_t cap; } TocCtx;

static int toc_visit(const char *path, void *ud) {
    TocCtx *ctx = (TocCtx *)ud;
    if (ctx->cap > 0 && ctx->count >= ctx->cap) {
        sb_append_str(ctx->out, "[... entry cap reached; use docs_search to find specific paths]\n");
        return 1;  /* stop walk */
    }

    size_t flen = 0;
    char *head = kb_read_file(path, &flen);
    char title_buf[256] = "(untitled)";
    const char *title = title_buf;
    if (head) {
        KbFrontmatter fm;
        kb_parse_frontmatter(head, flen > 4096 ? 4096 : flen, &fm);
        const char *fm_title = kb_fm_get(&fm, "title");
        if (fm_title && *fm_title) {
            size_t tl = strlen(fm_title);
            if (tl >= sizeof(title_buf)) tl = sizeof(title_buf) - 1;
            memcpy(title_buf, fm_title, tl);
            title_buf[tl] = '\0';
        } else {
            /* Fall back to first H1. */
            const char *body = head + fm.body_offset;
            const char *end = head + flen;
            while (body < end && (*body == '\n' || *body == '\r' || *body == ' ' || *body == '\t')) body++;
            if (body + 2 < end && body[0] == '#' && body[1] == ' ') {
                const char *h1 = body + 2;
                const char *eol = h1;
                while (eol < end && *eol != '\n') eol++;
                size_t hlen = eol - h1;
                if (hlen >= sizeof(title_buf)) hlen = sizeof(title_buf) - 1;
                memcpy(title_buf, h1, hlen);
                title_buf[hlen] = '\0';
            }
        }
        kb_fm_free(&fm);
        free(head);
    }

    sb_append_str(ctx->out, kb_strip_know_prefix(path));
    sb_append_str(ctx->out, ": ");
    sb_append_str(ctx->out, title);
    sb_append_char(ctx->out, '\n');
    ctx->count++;
    return 0;
}

char *execute_docs_toc(const char *args) {
    (void)args;
    StringBuf out;
    sb_init(&out);
    TocCtx ctx = { &out, 0, 200 };  /* hard cap of 200 entries */
    /* Walk in precedence order (notes > pinned > docs). */
    kb_walk_dir(KB_NOTES_DIR,  toc_visit, &ctx);
    kb_walk_dir(KB_PINNED_DIR, toc_visit, &ctx);
    kb_walk_dir(KB_DOCS_DIR,   toc_visit, &ctx);
    if (out.len == 0) {
        sb_free(&out);
        return strdup("(empty knowledge base. Add files via 'basi-cli docs add <path>' or '/note ...'.)\n");
    }
    return sb_to_str(&out);
}

/* ─── docs_get ─────────────────────────────────────────────────────── */

/* Find heading in [body, body+body_len) whose text equals `target` (case-
 * insensitive, ignoring trailing #s). Sets *out_start to the heading line
 * and *out_level to its level. Returns true on hit. */
static bool kb_find_heading(const char *body, size_t body_len,
                            const char *target,
                            const char **out_start, int *out_level) {
    size_t tlen = strlen(target);
    const char *p = body;
    const char *end = body + body_len;
    while (p < end) {
        const char *le = p;
        while (le < end && *le != '\n') le++;
        if (*p == '#') {
            int level = 0;
            while (level < 6 && p[level] == '#') level++;
            if (level < 6 && p[level] == ' ') {
                const char *h = p + level + 1;
                while (h < le && (*h == ' ' || *h == '\t')) h++;
                size_t hlen = le - h;
                while (hlen > 0 && (h[hlen-1] == ' ' || h[hlen-1] == '\t' ||
                                    h[hlen-1] == '#' || h[hlen-1] == '\r')) hlen--;
                if (hlen == tlen) {
                    bool eq = true;
                    for (size_t i = 0; i < hlen; i++) {
                        char a = h[i], b = target[i];
                        if (a >= 'A' && a <= 'Z') a += 32;
                        if (b >= 'A' && b <= 'Z') b += 32;
                        if (a != b) { eq = false; break; }
                    }
                    if (eq) { *out_start = p; *out_level = level; return true; }
                }
            }
        }
        p = (le < end) ? le + 1 : end;
    }
    return false;
}

/* List all H2 + H3 headings from body into `out` (one per line). */
static void kb_list_headings(const char *body, size_t body_len, StringBuf *out) {
    const char *p = body;
    const char *end = body + body_len;
    while (p < end) {
        const char *le = p;
        while (le < end && *le != '\n') le++;
        if (*p == '#') {
            int level = 0;
            while (level < 6 && p[level] == '#') level++;
            if ((level == 2 || level == 3) && p[level] == ' ') {
                const char *h = p + level + 1;
                while (h < le && (*h == ' ' || *h == '\t')) h++;
                size_t hlen = le - h;
                while (hlen > 0 && (h[hlen-1] == ' ' || h[hlen-1] == '\t' ||
                                    h[hlen-1] == '#' || h[hlen-1] == '\r')) hlen--;
                sb_append_str(out, "  - ");
                sb_append(out, h, hlen);
                sb_append_char(out, '\n');
            }
        }
        p = (le < end) ? le + 1 : end;
    }
}

char *execute_docs_get(const char *args) {
    while (*args == ' ' || *args == '\t') args++;
    if (!*args) {
        return strdup("docs_get not allowed: missing path. Example: docs_get docs/godot-4.4/classes/Node.md");
    }
    /* Split off optional #anchor and trim trailing whitespace from path. */
    const char *hash = strchr(args, '#');
    const char *anchor = hash ? hash + 1 : NULL;
    size_t plen = hash ? (size_t)(hash - args) : strlen(args);
    while (plen > 0 && (args[plen-1] == ' ' || args[plen-1] == '\t' || args[plen-1] == '\n')) plen--;
    if (plen == 0 || plen >= 1024) {
        return strdup("docs_get not allowed: path empty or too long.");
    }
    char path[1024];
    memcpy(path, args, plen);
    path[plen] = '\0';
    if (path[0] == '/' || strstr(path, "..") != NULL) {
        return strdup("docs_get not allowed: path must be relative under .basi/knowledge/ and may not contain '..'.");
    }

    char full[1280];
    snprintf(full, sizeof(full), "%s/%s", KB_KNOW_DIR, path);
    size_t file_len = 0;
    char *buf = kb_read_file(full, &file_len);
    if (!buf) {
        char *msg = malloc(512);
        snprintf(msg, 512, "docs_get not allowed: file not found at '%.400s'. Use docs_toc to list available paths.", path);
        return msg;
    }

    KbFrontmatter fm;
    kb_parse_frontmatter(buf, file_len, &fm);
    const char *body = buf + fm.body_offset;
    size_t body_len = file_len - fm.body_offset;
    kb_fm_free(&fm);

    StringBuf out;
    sb_init(&out);

    if (anchor && *anchor) {
        size_t alen = strlen(anchor);
        while (alen > 0 && (anchor[alen-1] == ' ' || anchor[alen-1] == '\t' ||
                            anchor[alen-1] == '\n' || anchor[alen-1] == '\r')) alen--;
        if (alen == 0 || alen >= 256) {
            free(buf);
            sb_free(&out);
            return strdup("docs_get not allowed: anchor empty or too long.");
        }
        char anchor_buf[256];
        memcpy(anchor_buf, anchor, alen);
        anchor_buf[alen] = '\0';
        /* Anchor may be "x/y/z"; we match against the leaf segment. */
        const char *leaf = strrchr(anchor_buf, '/');
        leaf = leaf ? leaf + 1 : anchor_buf;

        const char *match_start = NULL;
        int match_level = 0;
        if (!kb_find_heading(body, body_len, leaf, &match_start, &match_level)) {
            sb_append_str(&out, "docs_get not allowed: anchor '");
            sb_append_str(&out, anchor_buf);
            sb_append_str(&out, "' not found in '");
            sb_append_str(&out, path);
            sb_append_str(&out, "'. Available H2/H3 headings:\n");
            kb_list_headings(body, body_len, &out);
            free(buf);
            return sb_to_str(&out);
        }
        /* Find section end: next heading at same or higher level. */
        const char *section_end = body + body_len;
        const char *q = match_start;
        const char *qend = body + body_len;
        const char *q_le = q;
        while (q_le < qend && *q_le != '\n') q_le++;
        q = (q_le < qend) ? q_le + 1 : qend;
        while (q < qend) {
            const char *le = q;
            while (le < qend && *le != '\n') le++;
            if (*q == '#') {
                int lvl = 0;
                while (lvl < 6 && q[lvl] == '#') lvl++;
                if (lvl > 0 && lvl <= match_level && q[lvl] == ' ') {
                    section_end = q;
                    break;
                }
            }
            q = (le < qend) ? le + 1 : qend;
        }
        size_t slen = section_end - match_start;
        const size_t SECTION_CAP = 4096;
        if (slen <= SECTION_CAP) {
            sb_append(&out, match_start, slen);
        } else {
            sb_append(&out, match_start, SECTION_CAP);
            sb_append_str(&out, "\n[... section truncated. Drill in with a more specific anchor.]\n");
        }
    } else {
        const size_t CAP = 4096;
        if (body_len <= CAP) {
            sb_append(&out, body, body_len);
        } else {
            sb_append(&out, body, CAP);
            char tail[256];
            snprintf(tail, sizeof(tail),
                "\n\n[... truncated; full body is %zu bytes. Anchors available:]\n",
                body_len);
            sb_append_str(&out, tail);
            kb_list_headings(body, body_len, &out);
            sb_append_str(&out, "Call docs_get ");
            sb_append_str(&out, path);
            sb_append_str(&out, "#<anchor> to drill in.\n");
        }
    }

    free(buf);
    return sb_to_str(&out);
}

/* ─── docs_search ──────────────────────────────────────────────────── */

char *execute_docs_search(const char *args) {
    while (*args == ' ' || *args == '\t') args++;
    if (!*args) {
        return strdup("docs_search not allowed: missing keyword. Example: docs_search queue_free");
    }
    /* Trim trailing whitespace. */
    size_t qlen = strlen(args);
    while (qlen > 0 && (args[qlen-1] == ' ' || args[qlen-1] == '\t' ||
                        args[qlen-1] == '\n' || args[qlen-1] == '\r')) qlen--;
    if (qlen == 0 || qlen >= 256) {
        return strdup("docs_search not allowed: keyword empty or too long.");
    }
    /* Reject metacharacters that would let the model run arbitrary shell. */
    for (size_t i = 0; i < qlen; i++) {
        char c = args[i];
        if (c == '\'' || c == '`' || c == '$' || c == '\\' || c == '\n') {
            return strdup("docs_search not allowed: keyword may not contain quotes, backticks, $, backslash, or newline.");
        }
    }
    char query[256];
    memcpy(query, args, qlen);
    query[qlen] = '\0';

    /* Use system grep -rnF (literal, recursive, line numbers).
     * -F is fixed-string match — matches the spec ("literal substring only"). */
    StringBuf cmd;
    sb_init(&cmd);
    sb_append_str(&cmd, "grep -rnF --include='*.md' --color=never -- '");
    sb_append_str(&cmd, query);
    sb_append_str(&cmd, "' " KB_KNOW_DIR " 2>/dev/null | head -n 50");

    char *result = run_command(sb_to_str(&cmd), 64 * 1024);
    sb_free(&cmd);

    if (!result || !*result) {
        free(result);
        char *msg = malloc(256);
        snprintf(msg, 256, "docs_search: no matches for '%.180s' in knowledge base.\n", query);
        return msg;
    }

    /* Strip the ".basi/knowledge/" prefix from each line for readability. */
    StringBuf out;
    sb_init(&out);
    const char *p = result;
    const char *end = result + strlen(result);
    while (p < end) {
        const char *le = p;
        while (le < end && *le != '\n') le++;
        const char *trimmed = p;
        size_t pre = strlen(KB_KNOW_DIR) + 1;
        if ((size_t)(le - p) > pre &&
            strncmp(p, KB_KNOW_DIR "/", pre) == 0) trimmed = p + pre;
        sb_append(&out, trimmed, le - trimmed);
        if (le < end) sb_append_char(&out, '\n');
        p = (le < end) ? le + 1 : end;
    }
    free(result);
    return sb_to_str(&out);
}

/* ─── docs_recent_notes ────────────────────────────────────────────── */

typedef struct { char *path; time_t mtime; } NoteEntry;

static int note_cmp_desc(const void *a, const void *b) {
    const NoteEntry *na = (const NoteEntry *)a;
    const NoteEntry *nb = (const NoteEntry *)b;
    if (na->mtime > nb->mtime) return -1;
    if (na->mtime < nb->mtime) return 1;
    return 0;
}

char *execute_docs_recent_notes(const char *args) {
    (void)args;
    DIR *d = opendir(KB_NOTES_DIR);
    if (!d) {
        return strdup("(no notes yet — add one with '/note <text>'.)\n");
    }
    NoteEntry *entries = NULL;
    size_t n = 0, cap = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        if (de->d_name[0] == '.') continue;
        size_t nl = strlen(de->d_name);
        if (nl <= 3 || strcmp(de->d_name + nl - 3, ".md") != 0) continue;
        char full[1024];
        if ((size_t)snprintf(full, sizeof(full), "%s/%s", KB_NOTES_DIR, de->d_name) >= sizeof(full)) continue;
        struct stat st;
        if (stat(full, &st) != 0 || !S_ISREG(st.st_mode)) continue;
        if (n == cap) {
            cap = cap ? cap * 2 : 8;
            entries = realloc(entries, cap * sizeof(NoteEntry));
        }
        entries[n].path = strdup(full);
        entries[n].mtime = st.st_mtime;
        n++;
    }
    closedir(d);

    if (n == 0) {
        free(entries);
        return strdup("(no notes yet — add one with '/note <text>'.)\n");
    }
    qsort(entries, n, sizeof(NoteEntry), note_cmp_desc);

    StringBuf out;
    sb_init(&out);
    size_t total_lines = 0;
    const size_t LINE_CAP = 30;
    for (size_t i = 0; i < n && total_lines < LINE_CAP; i++) {
        size_t flen;
        char *body = kb_read_file(entries[i].path, &flen);
        if (!body) continue;
        KbFrontmatter fm;
        kb_parse_frontmatter(body, flen, &fm);
        const char *p = body + fm.body_offset;
        const char *end = body + flen;
        kb_fm_free(&fm);
        const char *rel = kb_strip_know_prefix(entries[i].path);
        while (p < end && total_lines < LINE_CAP) {
            const char *le = p;
            while (le < end && *le != '\n') le++;
            /* Skip blank lines. */
            const char *q = p;
            while (q < le && (*q == ' ' || *q == '\t')) q++;
            if (q < le) {
                sb_append_str(&out, "[");
                sb_append_str(&out, rel);
                sb_append_str(&out, "] ");
                sb_append(&out, p, le - p);
                sb_append_char(&out, '\n');
                total_lines++;
            }
            p = (le < end) ? le + 1 : end;
        }
        free(body);
    }
    if (total_lines == LINE_CAP) {
        sb_append_str(&out, "[... cap reached; older notes elided]\n");
    }

    for (size_t i = 0; i < n; i++) free(entries[i].path);
    free(entries);

    return sb_to_str(&out);
}

int cmd_docs_add(int argc, char **argv) {
    /* argv: [0]=basi-cli, [1]=docs, [2]=add, [3..]=args */
    const char *path = NULL;
    KbShelf shelf = KB_SHELF_PINNED;
    for (int i = 3; i < argc; i++) {
        if (strncmp(argv[i], "--shelf=", 8) == 0) {
            shelf = kb_shelf_from_name(argv[i] + 8);
            if (shelf == KB_SHELF_UNKNOWN) {
                fprintf(stderr, "docs add not allowed: --shelf must be notes|pinned|docs\n");
                return 2;
            }
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "docs add not allowed: unknown flag '%s'\n", argv[i]);
            return 2;
        } else if (!path) {
            path = argv[i];
        } else {
            fprintf(stderr, "docs add not allowed: only one path per invocation\n");
            return 2;
        }
    }
    if (!path) {
        fprintf(stderr, "Usage: basi-cli docs add <file.md> [--shelf=notes|pinned|docs]\n");
        return 2;
    }

    struct stat st;
    if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) {
        fprintf(stderr, "docs add not allowed: file not found: %s\n", path);
        return 1;
    }
    size_t plen = strlen(path);
    bool is_md = (plen >= 3 && strcmp(path + plen - 3, ".md") == 0) ||
                 (plen >= 9 && strcmp(path + plen - 9, ".markdown") == 0);
    if (!is_md) {
        fprintf(stderr, "docs add not allowed: only .md/.markdown supported in v1\n");
        return 1;
    }

    if (kb_ensure_dirs() != 0) {
        fprintf(stderr, "docs add not allowed: cannot create .basi/knowledge/ tree (%s)\n",
                strerror(errno));
        return 1;
    }

    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    char dest[1280];
    if ((size_t)snprintf(dest, sizeof(dest), "%s/%s", kb_shelf_dir(shelf), base) >= sizeof(dest)) {
        fprintf(stderr, "docs add not allowed: destination path too long\n");
        return 1;
    }
    if (stat(dest, &st) == 0) {
        fprintf(stderr, "docs add not allowed: already exists at %s (delete it first to re-add)\n", dest);
        return 1;
    }

    size_t src_len = 0;
    char *src = kb_read_file(path, &src_len);
    if (!src) {
        fprintf(stderr, "docs add not allowed: cannot read %s\n", path);
        return 1;
    }

    KbFrontmatter fm;
    if (kb_parse_frontmatter(src, src_len, &fm) < 0) {
        fprintf(stderr, "docs add not allowed: malformed YAML frontmatter in %s\n", path);
        free(src);
        return 1;
    }

    /* Title: existing frontmatter > first H1 > filename without ext. */
    char title_buf[256] = "";
    const char *fm_title = kb_fm_get(&fm, "title");
    if (fm_title) {
        size_t tl = strlen(fm_title);
        if (tl >= sizeof(title_buf)) tl = sizeof(title_buf) - 1;
        memcpy(title_buf, fm_title, tl);
        title_buf[tl] = '\0';
    } else {
        const char *body = src + fm.body_offset;
        const char *end = src + src_len;
        while (body < end && (*body == '\n' || *body == ' ' || *body == '\t' || *body == '\r')) body++;
        if (body + 2 < end && body[0] == '#' && body[1] == ' ') {
            const char *h = body + 2;
            const char *eol = h;
            while (eol < end && *eol != '\n') eol++;
            size_t hlen = eol - h;
            if (hlen >= sizeof(title_buf)) hlen = sizeof(title_buf) - 1;
            memcpy(title_buf, h, hlen);
            title_buf[hlen] = '\0';
        } else {
            size_t bl = strlen(base);
            if (bl > 3 && strcmp(base + bl - 3, ".md") == 0) bl -= 3;
            if (bl >= sizeof(title_buf)) bl = sizeof(title_buf) - 1;
            memcpy(title_buf, base, bl);
            title_buf[bl] = '\0';
        }
    }

    StringBuf out;
    sb_init(&out);
    sb_append_str(&out, "---\nsource: ");
    sb_append_str(&out, path);
    sb_append_str(&out, "\nshelf: ");
    sb_append_str(&out, kb_shelf_name(shelf));
    sb_append_str(&out, "\ntitle: ");
    sb_append_str(&out, title_buf);
    time_t now = time(NULL);
    struct tm tm_now;
    localtime_r(&now, &tm_now);
    char date_buf[32];
    strftime(date_buf, sizeof(date_buf), "%Y-%m-%d", &tm_now);
    sb_append_str(&out, "\nfetched: ");
    sb_append_str(&out, date_buf);
    sb_append_str(&out, "\n---\n");
    sb_append(&out, src + fm.body_offset, src_len - fm.body_offset);

    kb_fm_free(&fm);
    free(src);

    FILE *df = fopen(dest, "w");
    if (!df) {
        fprintf(stderr, "docs add not allowed: cannot create %s (%s)\n", dest, strerror(errno));
        sb_free(&out);
        return 1;
    }
    fwrite(out.data, 1, out.len, df);
    fclose(df);
    sb_free(&out);

    printf("[added] %s -> %s (shelf=%s, title=\"%s\")\n",
           path, dest, kb_shelf_name(shelf), title_buf);
    return 0;
}
