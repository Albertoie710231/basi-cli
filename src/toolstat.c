#include <stdlib.h>
#include <string.h>

#include "toolstat.h"

/* One row per tool NAME seen this run. A run touches a handful of distinct
 * tools, so a small fixed table beats a hash: no allocation on the dispatch
 * path, and overflow degrades to "not counted" rather than to a crash. */
#define TOOLSTAT_MAX   32
#define TOOLSTAT_NAME  64
#define TOOLSTAT_WHY  200

typedef struct {
    char name[TOOLSTAT_NAME];
    char why[TOOLSTAT_WHY];   /* first failure reason; "" while healthy */
    int  calls;
    int  fails;
} ToolStat;

static ToolStat rows[TOOLSTAT_MAX];
static int      rows_n = 0;

/* Find `tool`, or claim a row for it. NULL once the table is full — every caller
 * treats that as "stop counting", never as an error worth surfacing: losing a
 * statistic must not break the tool call it describes. */
static ToolStat *row_for(const char *tool) {
    if (!tool || !*tool) return NULL;
    for (int i = 0; i < rows_n; i++)
        if (strcmp(rows[i].name, tool) == 0) return &rows[i];
    if (rows_n >= TOOLSTAT_MAX) return NULL;

    ToolStat *r = &rows[rows_n++];
    snprintf(r->name, sizeof(r->name), "%s", tool);
    r->why[0] = '\0';
    r->calls = r->fails = 0;
    return r;
}

void basi_toolstat_call(const char *tool) {
    ToolStat *r = row_for(tool);
    if (r) r->calls++;
}

void basi_toolstat_fail(const char *tool, const char *reason) {
    ToolStat *r = row_for(tool);
    if (!r) return;
    r->fails++;
    /* A tool can fail on a path that never reached the central counter. Better a
     * conservative denominator than a report that says "1 of 0 calls failed". */
    if (r->fails > r->calls) r->calls = r->fails;
    /* Keep the FIRST reason: a backend that goes down stays down, and the first
     * failure is the one with the original cause rather than a knock-on. */
    if (!r->why[0] && reason && *reason)
        snprintf(r->why, sizeof(r->why), "%s", reason);
}

char *basi_toolstat_failed(const char *tool, const char *reason) {
    basi_toolstat_fail(tool, reason);

    const char *why = (reason && *reason) ? reason : "the backend was unreachable";
    size_t n = sizeof(BASI_TOOL_ERROR) + strlen(why);
    char *s = malloc(n);
    if (!s) return NULL;
    snprintf(s, n, BASI_TOOL_ERROR "%s", why);
    return s;
}

bool basi_toolstat_any_failed(void) {
    for (int i = 0; i < rows_n; i++)
        if (rows[i].fails > 0) return true;
    return false;
}

void basi_toolstat_report(FILE *f) {
    if (!f || !basi_toolstat_any_failed()) return;

    for (int i = 0; i < rows_n; i++) {
        const ToolStat *r = &rows[i];
        if (r->fails <= 0) continue;
        /* Yellow, on stderr, and phrased as a share of the calls: "1 of 9" is a
         * hiccup worth ignoring, "9 of 9" means the answer above was written
         * without a capability it thought it had. */
        fprintf(f, "\033[33m[degraded] %s: %d of %d call%s failed",
                r->name, r->fails, r->calls, r->calls == 1 ? "" : "s");
        if (r->why[0]) fprintf(f, " — %s", r->why);
        fprintf(f, "\033[0m\n");
    }
    fprintf(f, "\033[33m[degraded] the answer above was produced WITHOUT that tool"
               " — judge it accordingly.\033[0m\n");
}

bool basi_toolstat_row(int i, const char **name, int *calls, int *fails) {
    if (i < 0 || i >= rows_n) return false;
    if (name)  *name  = rows[i].name;
    if (calls) *calls = rows[i].calls;
    if (fails) *fails = rows[i].fails;
    return true;
}
