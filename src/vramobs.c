/* vramobs.c — measured VRAM per (model, ngl, ctx). See vramobs.h. */
#include "vramobs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* Bounded so the file can't grow without limit; newest entries are kept. */
#define VRAMOBS_MAX 64

typedef struct {
    double predicted_mb;
    double observed_mb;
    int    ngl;
    int    ctx;
    char   model[512];
} Obs;

static void obs_path(char *out, size_t n) {
    const char *xdg  = getenv("XDG_CONFIG_HOME");
    const char *home = getenv("HOME");
    if (xdg && *xdg)        snprintf(out, n, "%s/basi-cli/vram-observed", xdg);
    else if (home && *home) snprintf(out, n, "%s/.config/basi-cli/vram-observed", home);
    else                    snprintf(out, n, ".basi/vram-observed");
}

/* Load the file, newest first. Returns the count. */
static int obs_load(Obs *out, int max) {
    char path[600]; obs_path(path, sizeof path);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int n = 0;
    char line[900];
    /* Format: <predicted_mb> <observed_mb> <ngl> <ctx> <model path with spaces> */
    while (n < max && fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        Obs o; memset(&o, 0, sizeof o);
        int consumed = 0;
        if (sscanf(line, "%lf %lf %d %d %n",
                   &o.predicted_mb, &o.observed_mb, &o.ngl, &o.ctx, &consumed) != 4) continue;
        if (consumed <= 0) continue;
        char *m = line + consumed;
        size_t l = strlen(m);
        while (l > 0 && (m[l-1] == '\n' || m[l-1] == '\r' || m[l-1] == ' ')) m[--l] = '\0';
        if (!*m) continue;
        snprintf(o.model, sizeof o.model, "%s", m);
        out[n++] = o;
    }
    fclose(f);
    return n;
}

int vramobs_get(const char *model_path, int ngl, int ctx,
                double *predicted_mb, double *observed_mb) {
    if (!model_path || !*model_path) return 0;
    Obs list[VRAMOBS_MAX];
    int n = obs_load(list, VRAMOBS_MAX);
    for (int i = 0; i < n; i++) {
        if (list[i].ngl == ngl && list[i].ctx == ctx &&
            strcmp(list[i].model, model_path) == 0) {
            if (predicted_mb) *predicted_mb = list[i].predicted_mb;
            if (observed_mb)  *observed_mb  = list[i].observed_mb;
            return 1;
        }
    }
    return 0;
}

int vramobs_offset(const char *model_path, double *offset_mb) {
    if (!model_path || !*model_path) return 0;
    Obs list[VRAMOBS_MAX];
    int n = obs_load(list, VRAMOBS_MAX);
    for (int i = 0; i < n; i++) {          /* newest first */
        if (strcmp(list[i].model, model_path) == 0) {
            if (offset_mb) *offset_mb = list[i].observed_mb - list[i].predicted_mb;
            return 1;
        }
    }
    return 0;
}

void vramobs_record(const char *model_path, int ngl, int ctx,
                    double predicted_mb, double observed_mb) {
    if (!model_path || !*model_path) return;
    if (observed_mb <= 0) return;          /* nothing measurable — don't poison the store */

    Obs list[VRAMOBS_MAX];
    int n = obs_load(list, VRAMOBS_MAX);

    char path[600]; obs_path(path, sizeof path);
    /* Create the config dir if this is the first thing written there. */
    char dir[600]; snprintf(dir, sizeof dir, "%s", path);
    char *slash = strrchr(dir, '/');
    if (slash) { *slash = '\0'; if (dir[0]) mkdir(dir, 0755); }

    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "# BASI-CLI measured VRAM. predicted_mb observed_mb ngl ctx model\n");
    fprintf(f, "# Delete a line to force re-measurement on the next launch.\n");
    /* Newest first, so vramobs_offset() prefers the most recent measurement. */
    fprintf(f, "%.1f %.1f %d %d %s\n", predicted_mb, observed_mb, ngl, ctx, model_path);
    int written = 1;
    for (int i = 0; i < n && written < VRAMOBS_MAX; i++) {
        if (list[i].ngl == ngl && list[i].ctx == ctx &&
            strcmp(list[i].model, model_path) == 0) continue;   /* superseded */
        fprintf(f, "%.1f %.1f %d %d %s\n", list[i].predicted_mb, list[i].observed_mb,
                list[i].ngl, list[i].ctx, list[i].model);
        written++;
    }
    fclose(f);
}

double vramobs_correct(const char *model_path, int ngl, int ctx,
                       double predicted_mb, int *was_measured) {
    if (was_measured) *was_measured = 0;
    double obs = 0, pred = 0;
    if (vramobs_get(model_path, ngl, ctx, &pred, &obs)) {
        if (was_measured) *was_measured = 1;
        return obs;
    }
    double off = 0;
    if (vramobs_offset(model_path, &off)) {
        double v = predicted_mb + off;
        return v > 0 ? v : predicted_mb;   /* a bad offset must not predict negative */
    }
    return predicted_mb;
}
