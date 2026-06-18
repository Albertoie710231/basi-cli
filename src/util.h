#ifndef BASI_UTIL_H
#define BASI_UTIL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

/* ── Dynamic string buffer ─────────────────────────────────────────── */
typedef struct {
    char  *data;
    size_t len;
    size_t cap;
} StringBuf;

void  sb_init(StringBuf *sb);
void  sb_free(StringBuf *sb);
void  sb_ensure(StringBuf *sb, size_t extra);
void  sb_append(StringBuf *sb, const char *s, size_t n);
void  sb_append_str(StringBuf *sb, const char *s);
void  sb_append_char(StringBuf *sb, char c);
char *sb_to_str(StringBuf *sb);
void  sb_clear(StringBuf *sb);

/* ── Tokenized command line (respects single/double quotes) ────────── */
typedef struct {
    char **args;
    int    count;
} ArgList;

ArgList tokenize_command(const char *cmd);
void    arglist_free(ArgList *al);

/* ── Process / FS / time utilities ─────────────────────────────────── */
char  *run_command(const char *cmd, size_t max_output);  /* malloc'd, caller frees */
int    mkdir_p(const char *path);                         /* recursive mkdir, idempotent */
size_t count_lines(FILE *f);                              /* leaves f rewound */
double time_now(void);                                    /* seconds since epoch (monotonic-ish) */

/* ── URL helpers ───────────────────────────────────────────────────── */
char *url_encode(const char *input);  /* malloc'd */
char *url_decode(const char *input);  /* malloc'd */

/* ── JSON helpers ──────────────────────────────────────────────────── */
void json_escape_into(StringBuf *sb, const char *s);

/* Minimal JSON path extractor. Paths use dots; numeric segments index arrays
   (e.g. "results.0.url"). Assumes well-formed JSON. */
char *jx_get_string(const char *json, const char *path);  /* malloc'd or NULL */
long  jx_get_int(const char *json, const char *path);     /* -1 if missing    */
bool  jx_has_key(const char *json, const char *key);

#endif /* BASI_UTIL_H */
