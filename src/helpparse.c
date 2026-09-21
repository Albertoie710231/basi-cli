/* helpparse.c — what flags does this llama-server accept? See helpparse.h. */
#include "helpparse.h"
#include "util.h"      /* run_command_status */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define HELP_FLAGS_MAX 512

struct HelpSpec {
    char  bin[512];
    off_t size;         /* (size, mtime) detect a rebuild of the same path */
    time_t mtime;
    char *flags[HELP_FLAGS_MAX];
    int   n_flags;
    int   valid;
};

/* One entry is enough: validation happens once per launch, for one binary. */
static HelpSpec g_cache;

static void spec_clear(HelpSpec *hs) {
    for (int i = 0; i < hs->n_flags; i++) free(hs->flags[i]);
    hs->n_flags = 0;
    hs->valid = 0;
}

static void spec_add(HelpSpec *hs, const char *flag, size_t len) {
    if (hs->n_flags >= HELP_FLAGS_MAX || len == 0) return;
    for (int i = 0; i < hs->n_flags; i++)
        if (strncmp(hs->flags[i], flag, len) == 0 && hs->flags[i][len] == '\0') return;
    char *c = malloc(len + 1);
    if (!c) return;
    memcpy(c, flag, len); c[len] = '\0';
    hs->flags[hs->n_flags++] = c;
}

/* Pull the flag aliases off one help line.
 *
 * A flag line starts at column 0 and reads:
 *     -t,    --threads N        number of CPU threads ...
 *     -h,    --help, --usage    print usage and exit
 *     -fa,   --flash-attn [on|off|auto]   set Flash Attention ...
 * so: walk whitespace/comma-separated tokens from the start, keep the ones that
 * begin with '-', and STOP at the first token that doesn't. That first non-dash
 * token is either the metavar or the start of the description, and everything
 * after it is prose — which is what keeps `{none,linear,yarn}` and
 * "priority : low(-1), normal(0)" from being mistaken for flags. */
static void parse_flag_line(HelpSpec *hs, const char *line) {
    const char *p = line;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == ',') p++;
        if (!*p || *p == '\n') break;
        if (*p != '-') break;                     /* metavar/description reached */
        const char *start = p;
        while (*p && *p != ' ' && *p != '\t' && *p != ',' && *p != '\n') p++;
        spec_add(hs, start, (size_t)(p - start));
    }
}

const HelpSpec *helpspec_for(const char *server_bin) {
    if (!server_bin || !*server_bin) return NULL;

    struct stat st;
    if (stat(server_bin, &st) != 0) return NULL;

    /* Cache hit: same path, same bytes, same mtime — a rebuild invalidates it. */
    if (g_cache.valid && strcmp(g_cache.bin, server_bin) == 0 &&
        g_cache.size == st.st_size && g_cache.mtime == st.st_mtime)
        return &g_cache;

    spec_clear(&g_cache);
    snprintf(g_cache.bin, sizeof g_cache.bin, "%s", server_bin);
    g_cache.size  = st.st_size;
    g_cache.mtime = st.st_mtime;

    char cmd[700];
    snprintf(cmd, sizeof cmd, "\"%s\" --help 2>&1", server_bin);
    int code = -1;
    char *out = run_command_status(cmd, 256 * 1024, &code);
    if (!out || code != 0) { free(out); return NULL; }

    /* Walk lines; only column 0 matters. Continuation lines (wrapped descriptions,
     * "(env: …)", "(default: …)") are indented ~40 columns, and section banners
     * look like "----- common params -----". */
    char *save = NULL;
    for (char *line = strtok_r(out, "\n", &save); line;
         line = strtok_r(NULL, "\n", &save)) {
        if (line[0] != '-') continue;
        if (strncmp(line, "-----", 5) == 0) continue;
        parse_flag_line(&g_cache, line);
    }
    free(out);

    if (g_cache.n_flags == 0) return NULL;   /* unparseable → say nothing */
    g_cache.valid = 1;
    return &g_cache;
}

int helpspec_has_flag(const HelpSpec *hs, const char *flag) {
    if (!hs || !hs->valid || !flag || !*flag) return 0;
    for (int i = 0; i < hs->n_flags; i++)
        if (strcmp(hs->flags[i], flag) == 0) return 1;
    return 0;
}

int helpspec_flag_count(const HelpSpec *hs) {
    return (hs && hs->valid) ? hs->n_flags : 0;
}

int helpspec_check_script(const char *script_path, const char *server_bin,
                          char **out, int max_out) {
    const HelpSpec *hs = helpspec_for(server_bin);
    if (!hs) return -1;

    FILE *f = fopen(script_path, "r");
    if (!f) return -1;

    int found = 0;
    char line[2048];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') continue;                   /* markers and doc comments */
        /* Tokens, respecting nothing fancier than whitespace: the generated script
         * quotes only paths, and a quoted path never starts with '-'. */
        char *save = NULL;
        for (char *tok = strtok_r(line, " \t\r\n", &save); tok;
             tok = strtok_r(NULL, " \t\r\n", &save)) {
            if (tok[0] != '-' || !tok[1]) continue;
            /* A negative number is a value, not a flag (e.g. "-ngl -1"). */
            if (tok[1] >= '0' && tok[1] <= '9') continue;
            if (helpspec_has_flag(hs, tok)) continue;
            if (found < max_out && out) out[found] = strdup(tok);
            found++;
        }
    }
    fclose(f);
    return found;
}
