#ifndef BASI_KB_H
#define BASI_KB_H

#include <stdbool.h>
#include <stddef.h>

#include "util.h"

/* Filesystem layout under the project's `.basi/` dir. See decision #2 in
 * .basi/plans/knowledge-base.md. */
#define KB_ROOT        ".basi"
#define KB_PLANS_DIR   ".basi/plans"
#define KB_KNOW_DIR    ".basi/knowledge"
#define KB_NOTES_DIR   ".basi/knowledge/notes"
#define KB_PINNED_DIR  ".basi/knowledge/pinned"
#define KB_DOCS_DIR    ".basi/knowledge/docs"

typedef enum {
    KB_SHELF_NOTES   = 0,
    KB_SHELF_PINNED  = 1,
    KB_SHELF_DOCS    = 2,
    KB_SHELF_UNKNOWN = -1
} KbShelf;

const char *kb_shelf_name(KbShelf s);
const char *kb_shelf_dir(KbShelf s);
KbShelf     kb_shelf_from_name(const char *s);

/* Idempotent: creates .basi/{plans,knowledge/{notes,pinned,docs}}/. */
int  kb_ensure_dirs(void);

/* Slug rules: lowercase letters + digits + single hyphens, 1-64 chars,
 * starts with letter, no leading/trailing/double hyphens. */
bool kb_slug_valid(const char *s);

/* ── Frontmatter parser (flat YAML key:value, terminated by ---) ───── */
typedef struct {
    char *key;
    char *value;
} KbFmEntry;

typedef struct {
    KbFmEntry *entries;
    size_t     count;
    size_t     cap;
    size_t     body_offset;  /* byte offset where body starts after closing --- */
} KbFrontmatter;

void        kb_fm_init(KbFrontmatter *fm);
void        kb_fm_free(KbFrontmatter *fm);
const char *kb_fm_get(const KbFrontmatter *fm, const char *key);
int         kb_parse_frontmatter(const char *text, size_t len, KbFrontmatter *out);

/* ── Filesystem helpers shared with plan.c and the test driver ─────── */
char       *kb_read_file(const char *path, size_t *out_len);  /* malloc'd */
const char *kb_strip_know_prefix(const char *path);

typedef int (*KbVisitor)(const char *path, void *ud);
int kb_walk_dir(const char *dir, KbVisitor visit, void *ud);

/* ── Tool entry points (read-only librarian) ───────────────────────── */
char *execute_docs_toc(const char *args);
char *execute_docs_get(const char *args);
char *execute_docs_search(const char *args);
char *execute_docs_recent_notes(const char *args);

/* ── CLI subcommand: basi-cli docs add <path> [--shelf=...] ────────── */
int cmd_docs_add(int argc, char **argv);

#endif /* BASI_KB_H */
