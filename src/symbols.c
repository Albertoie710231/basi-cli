#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include "symbols.h"
#include "util.h"

/* Cap the report so a huge file cannot flood the context. Counts are computed over
 * ALL symbols before truncation, so the totals stay correct even when the listing
 * is cut — the count is the thing the model most often actually needs. */
#define SYM_MAX_LISTED   300
#define SYM_MAX_BYTES    16000
#define SYM_SIG_CHARS    100

/* Single-quote a shell argument: close, escape, reopen. */
static void sq_append(StringBuf *sb, const char *arg) {
    sb_append_char(sb, '\'');
    for (const char *p = arg; *p; p++) {
        if (*p == '\'') sb_append_str(sb, "'\\''");
        else            sb_append_char(sb, *p);
    }
    sb_append_char(sb, '\'');
}

/* One `ctags -x` row: "<name> <kind> <line> <file> <source line...>". */
typedef struct { const char *name, *kind, *src; int line; size_t name_len, kind_len; } Row;

static int parse_row(char *line, Row *r) {
    char *p = line;
    #define FIELD(dst, dlen) \
        while (*p == ' ' || *p == '\t') p++; \
        dst = p; while (*p && *p != ' ' && *p != '\t') p++; \
        dlen = (size_t)(p - dst); if (!dlen) return 0;
    FIELD(r->name, r->name_len)
    FIELD(r->kind, r->kind_len)
    char *ln = p;
    while (*ln == ' ' || *ln == '\t') ln++;
    if (!isdigit((unsigned char)*ln)) return 0;
    r->line = atoi(ln);
    while (*p == ' ' || *p == '\t') p++;
    while (*p && *p != ' ' && *p != '\t') p++;      /* skip line number */
    while (*p == ' ' || *p == '\t') p++;
    while (*p && *p != ' ' && *p != '\t') p++;      /* skip file path   */
    while (*p == ' ' || *p == '\t') p++;
    r->src = p;
    #undef FIELD
    return 1;
}

/* Tally distinct kinds in first-seen order — no allocation, few kinds per file. */
#define MAX_KINDS 24
typedef struct { char name[32]; int n; } KindCount;

char *execute_symbols(const char *file, const char *kind) {
    if (!file || !*file) return strdup("Error: symbols requires a file path");
    if (access(file, R_OK) != 0) {
        StringBuf e; sb_init(&e);
        sb_append_str(&e, "Error: cannot read '");
        sb_append_str(&e, file);
        sb_append_str(&e, "'");
        return sb_to_str(&e);
    }
    if (kind && !*kind) kind = NULL;

    StringBuf cmd; sb_init(&cmd);
    /* No "--" here: universal-ctags rejects it as an unknown option. A path that
     * starts with '-' would otherwise be read as a flag, so relative-path leading
     * dashes get a "./" prefix instead. */
    sb_append_str(&cmd, "ctags -x --sort=no ");
    if (file[0] == '-') {
        StringBuf safe; sb_init(&safe);
        sb_append_str(&safe, "./");
        sb_append_str(&safe, file);
        sq_append(&cmd, sb_to_str(&safe));
        sb_free(&safe);
    } else {
        sq_append(&cmd, file);
    }
    sb_append_str(&cmd, " 2>&1");
    int timed_out = 0;
    char *out = run_command_timeout(sb_to_str(&cmd), 2u * 1024 * 1024, 20, &timed_out);
    sb_free(&cmd);

    if (timed_out) { free(out); return strdup("Error: ctags timed out after 20s"); }
    if (!out) return strdup("Error: failed to run ctags");
    if (strstr(out, "command not found") || strstr(out, "orden no encontrada")) {
        free(out);
        return strdup("Error: ctags is not installed. Install universal-ctags, or fall "
                      "back to reading the file directly — do NOT try to enumerate "
                      "definitions with grep regexes, it is unreliable.");
    }

    KindCount kinds[MAX_KINDS]; int n_kinds = 0;
    int total = 0, listed = 0, matched = 0;
    StringBuf body; sb_init(&body);

    for (char *line = strtok(out, "\n"); line; line = strtok(NULL, "\n")) {
        Row r;
        if (!parse_row(line, &r)) continue;
        total++;

        char kbuf[32];
        size_t kl = r.kind_len < sizeof kbuf - 1 ? r.kind_len : sizeof kbuf - 1;
        memcpy(kbuf, r.kind, kl); kbuf[kl] = '\0';

        int ki = -1;
        for (int i = 0; i < n_kinds; i++) if (strcmp(kinds[i].name, kbuf) == 0) { ki = i; break; }
        if (ki < 0 && n_kinds < MAX_KINDS) {
            ki = n_kinds++;
            snprintf(kinds[ki].name, sizeof kinds[ki].name, "%s", kbuf);
            kinds[ki].n = 0;
        }
        if (ki >= 0) kinds[ki].n++;

        if (kind && strcmp(kbuf, kind) != 0) continue;
        matched++;
        if (listed >= SYM_MAX_LISTED || body.len >= SYM_MAX_BYTES) continue;
        listed++;

        char nbuf[96];
        size_t nl = r.name_len < sizeof nbuf - 1 ? r.name_len : sizeof nbuf - 1;
        memcpy(nbuf, r.name, nl); nbuf[nl] = '\0';

        char head[220];
        snprintf(head, sizeof head, "%6d  %-10s %-28s ", r.line, kbuf, nbuf);
        sb_append_str(&body, head);
        for (const char *s = r.src; *s && (size_t)(s - r.src) < SYM_SIG_CHARS; s++)
            sb_append_char(&body, (*s == '\t') ? ' ' : *s);
        sb_append_char(&body, '\n');
    }
    free(out);

    StringBuf res; sb_init(&res);
    char hdr[512];
    if (total == 0) {
        snprintf(hdr, sizeof hdr,
                 "%s: ctags found no symbols (unsupported language, or the file "
                 "defines nothing at top level).\n", file);
        sb_append_str(&res, hdr);
        sb_free(&body);
        return sb_to_str(&res);
    }

    snprintf(hdr, sizeof hdr, "%s — %d symbols", file, total);
    sb_append_str(&res, hdr);
    for (int i = 0; i < n_kinds; i++) {
        char kb[64];   /* ": 72 function" — %.31s bounds it to KindCount.name[32] */
        snprintf(kb, sizeof kb, "%s %d %.31s", i == 0 ? ":" : ",", kinds[i].n, kinds[i].name);
        sb_append_str(&res, kb);
    }
    sb_append_str(&res, "\n");
    if (kind) {
        snprintf(hdr, sizeof hdr, "filtered to kind='%s': %d\n", kind, matched);
        sb_append_str(&res, hdr);
    }
    sb_append_str(&res, "\n  line  kind       name                         definition\n");
    sb_append_str(&res, sb_to_str(&body));
    if (matched > listed) {
        snprintf(hdr, sizeof hdr,
                 "\n[... %d more not shown; counts above are complete. Filter with "
                 "kind=, or read the file directly.]\n", matched - listed);
        sb_append_str(&res, hdr);
    }
    sb_free(&body);
    return sb_to_str(&res);
}
