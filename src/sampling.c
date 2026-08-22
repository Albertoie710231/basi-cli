#include "sampling.h"
#include "util.h"      /* jx_get_string */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>

static char *trim(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    char *e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) *--e = '\0';
    return s;
}

const char *sampling_config_path(void) {
    static char path[512];
    if (!path[0]) {
        const char *xdg  = getenv("XDG_CONFIG_HOME");
        const char *home = getenv("HOME");
        if (xdg && *xdg)        snprintf(path, sizeof path, "%s/basi-cli/sampling", xdg);
        else if (home && *home) snprintf(path, sizeof path, "%s/.config/basi-cli/sampling", home);
        else                    snprintf(path, sizeof path, ".basi/sampling");
    }
    return path;
}

static void profile_clear(SamplingProfile *p) {
    p->temperature = p->top_p = p->min_p = p->repeat_penalty = -1.0;
    p->top_k = -1;
    p->source[0] = p->match[0] = '\0';
}

/* Case-insensitive substring: does the model path contain this key? */
static bool path_matches(const char *path, const char *key) {
    if (!*key) return false;
    size_t kn = strlen(key);
    for (const char *p = path; *p; p++)
        if (strncasecmp(p, key, kn) == 0) return true;
    return false;
}

/* Apply one "name=value" token. Returns 1 if it set something. */
static int apply_kv(SamplingProfile *p, char *tok) {
    char *eq = strchr(tok, '=');
    if (!eq) return 0;
    *eq = '\0';
    char *k = trim(tok), *v = trim(eq + 1);
    if (!*k || !*v) return 0;
    if      (!strcasecmp(k, "temperature") || !strcasecmp(k, "temp")) p->temperature = atof(v);
    else if (!strcasecmp(k, "top_p")  || !strcasecmp(k, "top-p"))     p->top_p  = atof(v);
    else if (!strcasecmp(k, "top_k")  || !strcasecmp(k, "top-k"))     p->top_k  = atoi(v);
    else if (!strcasecmp(k, "min_p")  || !strcasecmp(k, "min-p"))     p->min_p  = atof(v);
    else if (!strcasecmp(k, "repeat_penalty") || !strcasecmp(k, "repetition_penalty"))
                                                                      p->repeat_penalty = atof(v);
    else return 0;
    return 1;
}

/* ~/.config/basi-cli/sampling, one line per model family:
 *     <key> = temperature=1.0 top_p=0.95 top_k=20 min_p=0.0 repeat_penalty=1.0
 * <key> is matched case-insensitively as a SUBSTRING of the model path, so
 * "qwen3.8" catches every quant and every directory layout. First match wins,
 * which makes ordering the way you express "specific before general". */
static int load_from_config(const char *model_path, SamplingProfile *out) {
    FILE *f = fopen(sampling_config_path(), "r");
    if (!f) return 0;
    char line[1024];
    int found = 0;
    while (fgets(line, sizeof line, f)) {
        char *s = trim(line);
        if (!*s || *s == '#') continue;
        char *eq = strchr(s, '=');
        if (!eq) continue;
        *eq = '\0';
        char *key = trim(s);
        if (!path_matches(model_path, key)) continue;

        snprintf(out->match, sizeof out->match, "%s", key);
        for (char *tok = strtok(eq + 1, " \t"); tok; tok = strtok(NULL, " \t"))
            found += apply_kv(out, tok);
        if (found) snprintf(out->source, sizeof out->source, "%s (%s)",
                            sampling_config_path(), out->match);
        break;                                   /* first match wins */
    }
    fclose(f);
    return found;
}

/* Read a JSON number by key. jx_get_int truncates and jx_get_string is for
 * strings, so temperature 1.0 / top_p 0.95 need their own reader. Deliberately
 * flat and dumb: generation_config.json is a one-level object of scalars. */
static int json_number(const char *json, const char *key, double *out) {
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return 0;
    p = strchr(p + strlen(pat), ':');
    if (!p) return 0;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n') p++;
    if (*p != '-' && *p != '+' && *p != '.' && !isdigit((unsigned char)*p)) return 0;
    *out = atof(p);
    return 1;
}

/* The vendor's own inference defaults, when the repo shipped them next to the
 * weights. A GGUF quant repo usually does NOT — it holds .gguf files and
 * nothing else — so this covers full-precision checkouts and the quant repos
 * thoughtful enough to include it, and quietly does nothing otherwise. */
static int load_from_generation_config(const char *model_path, SamplingProfile *out) {
    char dir[1024];
    snprintf(dir, sizeof dir, "%s", model_path);
    char *slash = strrchr(dir, '/');
    if (!slash) return 0;
    *slash = '\0';

    char path[1152];
    snprintf(path, sizeof path, "%s/generation_config.json", dir);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char buf[8192];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = '\0';

    int found = 0;
    double v;
    if (json_number(buf, "temperature", &v)) { out->temperature = v; found++; }
    if (json_number(buf, "top_p", &v))       { out->top_p       = v; found++; }
    if (json_number(buf, "top_k", &v))       { out->top_k = (int) v; found++; }
    if (found) snprintf(out->source, sizeof out->source, "%s", path);
    return found;
}

int sampling_profile_for(const char *model_path, SamplingProfile *out) {
    if (!out) return 0;
    profile_clear(out);
    if (!model_path || !*model_path) return 0;

    /* The declared config wins over the shipped JSON: it is the only source that
     * can carry min_p and the penalties, and it is the user saying explicitly
     * what this model should run with. */
    int n = load_from_config(model_path, out);
    if (n) return n;
    return load_from_generation_config(model_path, out);
}
