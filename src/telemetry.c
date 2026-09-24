#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "telemetry.h"
#include "toolstat.h"
#include "util.h"

/* Everything here is best-effort: a statistic that cannot be written must never
 * cost the user the turn it describes, so every failure is silent. */

static char ctx_backend[256] = "local";
static char ctx_model[256]   = "";
static char ctx_mode[16]     = "repl";
static char ctx_session[512] = "";

/* Per-turn tool deltas: toolstat counts for the whole run, so remember where the
 * previous turn left off. Same fixed-table bound as toolstat itself. */
#define TM_MAX 32
static struct { char name[64]; int calls, fails; } prev[TM_MAX];
static int    prev_n = 0;
static double turn_t0 = 0;

static void config_path(char *out, size_t n) {
    const char *xdg = getenv("XDG_CONFIG_HOME"), *home = getenv("HOME");
    if (xdg && *xdg)        snprintf(out, n, "%s/basi-cli/telemetry", xdg);
    else                    snprintf(out, n, "%s/.config/basi-cli/telemetry", home ? home : ".");
}

static void data_path(char *out, size_t n) {
    const char *xdg = getenv("XDG_DATA_HOME"), *home = getenv("HOME");
    if (xdg && *xdg)        snprintf(out, n, "%s/basi-cli/telemetry.jsonl", xdg);
    else                    snprintf(out, n, "%s/.local/share/basi-cli/telemetry.jsonl", home ? home : ".");
}

/* Saved choice: 1 on, 0 off, -1 never chosen (so the one-time notice is due). */
static int saved_choice(void) {
    char p[600]; config_path(p, sizeof p);
    FILE *f = fopen(p, "r");
    if (!f) return -1;
    char line[32] = "";
    if (!fgets(line, sizeof line, f)) line[0] = '\0';
    fclose(f);
    return strncmp(line, "off", 3) == 0 ? 0 : 1;
}

static bool save_choice(const char *v) {
    char p[600]; config_path(p, sizeof p);
    char dir[600]; snprintf(dir, sizeof dir, "%s", p);
    char *slash = strrchr(dir, '/');
    if (slash) { *slash = '\0'; mkdir_p(dir); }
    FILE *f = fopen(p, "w");
    if (!f) return false;
    fprintf(f, "%s\n", v);
    fclose(f);
    return true;
}

bool telemetry_enabled(void) {
    const char *e = getenv("BASI_TELEMETRY");
    if (e && *e) {
        if (strcmp(e, "0") == 0 || strcasecmp(e, "off") == 0 ||
            strcasecmp(e, "false") == 0 || strcasecmp(e, "no") == 0) return false;
        return true;
    }
    return saved_choice() != 0;
}

static void copy_str(char *dst, size_t n, const char *src) {
    size_t l = src ? strlen(src) : 0;
    if (l >= n) l = n - 1;
    if (l) memcpy(dst, src, l);
    dst[l] = '\0';
}

void telemetry_set_context(const char *backend, const char *model, const char *mode) {
    if (backend && *backend) copy_str(ctx_backend, sizeof ctx_backend, backend);
    if (model)               copy_str(ctx_model,   sizeof ctx_model,   model);
    if (mode && *mode)       copy_str(ctx_mode,    sizeof ctx_mode,    mode);
}

void telemetry_set_session(const char *session_path) {
    const char *base = session_path ? strrchr(session_path, '/') : NULL;
    copy_str(ctx_session, sizeof ctx_session, base ? base + 1 : session_path);
}

void telemetry_turn_begin(void) {
    turn_t0 = time_now();
}

static int prev_index(const char *name) {
    for (int i = 0; i < prev_n; i++)
        if (strcmp(prev[i].name, name) == 0) return i;
    if (prev_n >= TM_MAX) return -1;
    copy_str(prev[prev_n].name, sizeof prev[prev_n].name, name);
    prev[prev_n].calls = prev[prev_n].fails = 0;
    return prev_n++;
}

void telemetry_turn_end(const char *outcome, int rounds, int elision_resets,
                        size_t prompt_tokens, size_t gen_tokens, double gen_tps) {
    if (!telemetry_enabled()) return;

    /* Being clear about it is the point: the first record ever written says so,
       where it lives and how to stop it. Saving "on" is what marks it as said. */
    if (saved_choice() < 0 && !getenv("BASI_TELEMETRY")) {
        char dp[600]; data_path(dp, sizeof dp);
        fprintf(stderr,
            "\033[90m[telemetry] BASI keeps LOCAL usage stats (outcomes, rounds, tokens, "
            "tool call/failure counts — never prompts or output) in %s.\n"
            "[telemetry] Nothing leaves this machine. Turn it off: basi-cli telemetry off\033[0m\n",
            dp);
        save_choice("on");
    }

    char path[600]; data_path(path, sizeof path);
    {
        char dir[600]; snprintf(dir, sizeof dir, "%s", path);
        char *slash = strrchr(dir, '/');
        if (slash) { *slash = '\0'; mkdir_p(dir); }
    }

    char ts[32];
    { time_t now = time(NULL); strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%S%z", localtime(&now)); }
    char cwd[1024];
    if (!getcwd(cwd, sizeof cwd)) cwd[0] = '\0';

    StringBuf sb; sb_init(&sb);
    char num[128];
    sb_append_str(&sb, "{\"v\":1,\"ts\":");       json_escape_into(&sb, ts);
    sb_append_str(&sb, ",\"project\":");          json_escape_into(&sb, cwd);
    sb_append_str(&sb, ",\"backend\":");          json_escape_into(&sb, ctx_backend);
    sb_append_str(&sb, ",\"model\":");            json_escape_into(&sb, ctx_model);
    sb_append_str(&sb, ",\"mode\":");             json_escape_into(&sb, ctx_mode);
    if (ctx_session[0]) {
        sb_append_str(&sb, ",\"session\":");    json_escape_into(&sb, ctx_session);
    }
    {
        const char *tag = getenv("BASI_TELEMETRY_TAG");
        if (tag && *tag) { sb_append_str(&sb, ",\"tag\":"); json_escape_into(&sb, tag); }
    }
    sb_append_str(&sb, ",\"outcome\":");          json_escape_into(&sb, outcome ? outcome : "answered");
    snprintf(num, sizeof num,
             ",\"rounds\":%d,\"elision_resets\":%d,\"prompt_tokens\":%zu,"
             "\"gen_tokens\":%zu,\"gen_tps\":%.1f,\"secs\":%.1f",
             rounds, elision_resets, prompt_tokens, gen_tokens, gen_tps,
             turn_t0 > 0 ? time_now() - turn_t0 : 0.0);
    sb_append_str(&sb, num);

    /* Only the tools this turn touched, as deltas against the previous turn. */
    sb_append_str(&sb, ",\"tools\":{");
    bool first = true;
    const char *name; int calls, fails;
    for (int i = 0; basi_toolstat_row(i, &name, &calls, &fails); i++) {
        int k = prev_index(name);
        int dc = calls - (k >= 0 ? prev[k].calls : 0);
        int df = fails - (k >= 0 ? prev[k].fails : 0);
        if (k >= 0) { prev[k].calls = calls; prev[k].fails = fails; }
        if (dc <= 0 && df <= 0) continue;
        if (!first) sb_append_char(&sb, ',');
        first = false;
        json_escape_into(&sb, name);
        snprintf(num, sizeof num, ":{\"calls\":%d,\"fails\":%d}", dc, df);
        sb_append_str(&sb, num);
    }
    sb_append_str(&sb, "}}\n");

    char *line = sb_to_str(&sb);
    FILE *f = fopen(path, "a");
    if (f && line) { fputs(line, f); }
    if (f) fclose(f);
    free(line);
}

/* ── basi-cli telemetry ─────────────────────────────────────────────────── */

static void print_status(void) {
    char dp[600]; data_path(dp, sizeof dp);
    char cp[600]; config_path(cp, sizeof cp);
    const char *env = getenv("BASI_TELEMETRY");
    bool on = telemetry_enabled();

    printf("Local telemetry: %s", on ? "ON" : "OFF");
    if (env && *env)            printf("  (forced by BASI_TELEMETRY=%s)", env);
    else if (saved_choice() < 0) printf("  (default; not chosen yet)");
    printf("\n");

    struct stat st;
    long n = 0;
    FILE *f = fopen(dp, "r");
    if (f) { int c; while ((c = fgetc(f)) != EOF) if (c == '\n') n++; fclose(f); }
    if (stat(dp, &st) == 0)
        printf("  data:   %s  (%ld turn%s, %lld bytes)\n", dp, n, n == 1 ? "" : "s",
               (long long)st.st_size);
    else
        printf("  data:   %s  (nothing recorded yet)\n", dp);
    printf("  choice: %s\n", cp);
    printf("\n  Stays on this machine: no network, not used for training. Records\n"
           "  outcomes, rounds, tokens, speed and per-tool call/failure counts —\n"
           "  never prompts, answers, tool arguments or tool output.\n");
}

int telemetry_cmd(int argc, char **argv) {
    const char *sub = argc >= 3 ? argv[2] : "status";
    char dp[600]; data_path(dp, sizeof dp);

    if (strcmp(sub, "status") == 0) { print_status(); return 0; }
    if (strcmp(sub, "path") == 0)   { printf("%s\n", dp); return 0; }
    if (strcmp(sub, "on") == 0 || strcmp(sub, "off") == 0) {
        if (!save_choice(sub)) { fprintf(stderr, "Could not save the choice.\n"); return 1; }
        printf("Local telemetry %s.%s\n", strcmp(sub, "on") == 0 ? "ON" : "OFF",
               strcmp(sub, "off") == 0
                   ? " Existing data is kept; `basi-cli telemetry purge` deletes it." : "");
        return 0;
    }
    if (strcmp(sub, "purge") == 0) {
        if (unlink(dp) == 0)        printf("Deleted %s\n", dp);
        else if (errno == ENOENT)   printf("Nothing to delete.\n");
        else { perror(dp); return 1; }
        return 0;
    }
    if (strcmp(sub, "show") == 0) {
        long want = argc >= 4 ? atol(argv[3]) : 20;
        if (want <= 0) want = 20;
        size_t len = 0;
        char *all = read_file_all(dp, &len);
        if (!all) { printf("Nothing recorded yet.\n"); return 0; }
        /* Last `want` lines: walk back from the end. */
        const char *start = all;
        long seen = 0;
        for (size_t i = len; i > 0; i--) {
            if (all[i - 1] == '\n' && i != len && ++seen >= want) { start = all + i; break; }
        }
        fputs(start, stdout);
        free(all);
        return 0;
    }
    fprintf(stderr,
        "Usage:\n"
        "  basi-cli telemetry            show whether it is on and where the data is\n"
        "  basi-cli telemetry off|on     stop / resume recording (persists)\n"
        "  basi-cli telemetry show [N]   print the last N records (default 20)\n"
        "  basi-cli telemetry purge      delete everything recorded\n"
        "  basi-cli telemetry path       print the data file path\n"
        "  BASI_TELEMETRY=0 basi ...     off for a single run\n"
        "  BASI_TELEMETRY_TAG=A basi ... label this run's records (for A/B tests)\n");
    return 2;
}
