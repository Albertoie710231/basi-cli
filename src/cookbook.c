/* cookbook.c — /cookbook: discover, download, and manage local GGUF models.
 *
 * Ported from odysseus's "cookbook" (a Python/web feature that orchestrated HF
 * downloads + tmux serving + remote hosts). BASI runs models in-process via
 * llama.cpp, so the serve/tmux/remote machinery doesn't map. What ports cleanly
 * is the model lifecycle the user actually needs at the CLI:
 *
 *     /cookbook                list cached models + starter picks
 *     /cookbook search [q]     trending HF GGUF models that fit this box's VRAM
 *     /cookbook get <repo>     download a GGUF (curl + HF resolve URL)
 *     /cookbook rm <name>      delete a cached model
 *
 * Downloads are self-contained: curl straight from huggingface.co/<repo>/
 * resolve/main/<file> into ~/models (a dir BASI already scans), so no Python /
 * `hf` CLI dependency. `/model <name>` then switches to the freshly pulled file.
 */
#include "cookbook.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#include "util.h"     /* run_command, jx_get_string, jx_get_int, mkdir_p, url_encode */
#include "hwinfo.h"   /* hw_probe — VRAM budget for the "fits" filter */
#include "model.h"    /* basi_list_models — cached .gguf scan */

#define COOKBOOK_UA "basi-cli/cookbook"
#define HF_API "https://huggingface.co/api/models"

/* ── Starter picks ──────────────────────────────────────────────────────
 * A short curated catalog. These are validated live the moment you `get`
 * one (the download resolves against the real HF API), so a drifted id just
 * yields a clean error, never a crash. `/cookbook search` is the live, always-
 * current discovery path — this list is only a friendly on-ramp. */
typedef struct {
    const char *repo;
    const char *note;
} Preset;

static const Preset PRESETS[] = {
    { "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF",   "reasoning distill, 8B" },
    { "bartowski/Qwen2.5-7B-Instruct-GGUF",       "solid general-purpose, 7B" },
    { "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF","Llama 3.1 instruct, 8B" },
    { "bartowski/Mistral-7B-Instruct-v0.3-GGUF",  "compact + fast, 7B" },
};
#define N_PRESETS ((int)(sizeof(PRESETS) / sizeof(PRESETS[0])))

/* Quant preference when the user doesn't name one — good size/quality balance
 * first, then progressively larger/smaller fallbacks. */
static const char *QUANT_PREFS[] = {
    "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q4_0", "Q6_K", "Q8_0", "Q3_K_M", NULL
};

/* ── small helpers ──────────────────────────────────────────────────── */

/* Case-insensitive substring search (strcasestr isn't portable without
 * _GNU_SOURCE, and we'd rather not leak that define into this TU). */
static const char *ci_find(const char *hay, const char *needle) {
    if (!hay || !needle || !*needle) return hay;
    size_t nl = strlen(needle);
    for (const char *p = hay; *p; p++)
        if (strncasecmp(p, needle, nl) == 0) return p;
    return NULL;
}

static const char *home_dir(void) {
    const char *h = getenv("HOME");
    return (h && *h) ? h : ".";
}

/* Destination directory for downloads: $HOME/models — one of the dirs the
 * model picker already scans, so a pull is immediately switch-able. */
static void models_dir(char *out, size_t n) {
    snprintf(out, n, "%s/models", home_dir());
}

static const char *base_name(const char *path) {
    const char *s = strrchr(path, '/');
    return s ? s + 1 : path;
}

/* Human-readable byte count (decimal, matching HF/curl display). */
static void human_size(long bytes, char *out, size_t n) {
    if (bytes < 0)              snprintf(out, n, "?");
    else if (bytes >= 1000000000L) snprintf(out, n, "%.1f GB", bytes / 1e9);
    else if (bytes >= 1000000L)    snprintf(out, n, "%.0f MB", bytes / 1e6);
    else                       snprintf(out, n, "%ld KB", bytes / 1000);
}

/* A repo id must be exactly "org/name" over a safe charset — this string is
 * later interpolated into curl URLs and shell commands. */
static bool valid_repo(const char *s) {
    if (!s || !*s || s[0] == '/') return false;
    int slashes = 0;
    for (const char *p = s; *p; p++) {
        char c = *p;
        if (c == '/') { slashes++; continue; }
        if (!(isalnum((unsigned char)c) || c == '.' || c == '_' || c == '-'))
            return false;
    }
    return slashes == 1 && s[strlen(s) - 1] != '/';
}

/* Filenames come back from the HF API and get interpolated into a shell curl
 * command — accept only a conservative charset and no "..". */
static bool safe_rfilename(const char *s) {
    if (!s || !*s || s[0] == '/') return false;
    if (strstr(s, "..")) return false;
    for (const char *p = s; *p; p++) {
        char c = *p;
        if (!(isalnum((unsigned char)c) || c == '.' || c == '_' ||
              c == '-' || c == '/'))
            return false;
    }
    return true;
}

/* Rough parameter count parsed from a repo id, e.g. "...-7B-Instruct" -> 7.0,
 * "...-3.2-1B" -> 1.0. Returns 0 when no "<num>B" token is present. */
static double parse_params_b(const char *id) {
    for (const char *p = id; *p; p++) {
        if (!isdigit((unsigned char)*p)) continue;
        /* start of a number: reject if the previous char was a letter/digit so
         * we don't grab the "5" out of "v0.5" style version fragments mid-word */
        const char *q = p;
        double val = 0; int dot = 0;
        while (*q && (isdigit((unsigned char)*q) || (*q == '.' && !dot))) {
            if (*q == '.') { dot = 1; }
            q++;
        }
        /* number spans [p,q); the suffix must be a bare B/b */
        if ((*q == 'B' || *q == 'b') && !isalpha((unsigned char)q[1])) {
            char tmp[32];
            size_t len = (size_t)(q - p);
            if (len < sizeof tmp) {
                memcpy(tmp, p, len); tmp[len] = 0;
                val = atof(tmp);
                if (val > 0 && val < 2000) return val;
            }
        }
        p = q - 1;
    }
    return 0;
}

/* Approximate resident VRAM (GB) to run a model at ~Q4_K_M: weights are
 * ~0.6 bytes/param, plus a fixed slice for KV cache + compute scratch. */
static double est_gb_q4(double params_b) {
    if (params_b <= 0) return 0;
    return params_b * 0.6 + 0.8;
}

/* GET a URL, expecting JSON. Returns malloc'd body or NULL. */
static char *http_get_json(const char *url) {
    char cmd[1400];
    snprintf(cmd, sizeof cmd,
        "curl -sS --compressed -m 20 -A '" COOKBOOK_UA "' "
        "-H 'Accept: application/json' '%s' 2>/dev/null", url);
    char *body = run_command(cmd, 4 * 1024 * 1024);
    if (body && !body[0]) { free(body); return NULL; }
    return body;
}

/* Final Content-Length of a resolve URL (follows redirects to the CDN), or -1
 * if the server won't say. Used to show the size and to skip a completed file. */
static long remote_size(const char *url) {
    char cmd[1400];
    snprintf(cmd, sizeof cmd,
        "curl -sIL -m 25 -A '" COOKBOOK_UA "' '%s' 2>/dev/null", url);
    char *hdr = run_command(cmd, 256 * 1024);
    if (!hdr) return -1;
    long len = -1;
    const char *p = hdr;
    /* keep the LAST content-length seen (the final 200 after redirects) */
    while ((p = ci_find(p, "content-length:")) != NULL) {
        const char *v = p + strlen("content-length:");
        while (*v == ' ' || *v == '\t') v++;
        if (isdigit((unsigned char)*v)) len = atol(v);
        p += 1;
    }
    free(hdr);
    return len;
}

/* ── list ───────────────────────────────────────────────────────────── */

static void cmd_list(void) {
    printf("\n\033[1mCached models\033[0m \033[90m(what /model can switch to)\033[0m\n");
    char **models = NULL;
    int n = basi_list_models(&models);
    if (n == 0) {
        printf("  \033[90m(none yet — pull one with /cookbook get <org/name>)\033[0m\n");
    } else {
        for (int i = 0; i < n; i++) {
            struct stat st;
            char sz[24] = "?";
            if (stat(models[i], &st) == 0) human_size((long)st.st_size, sz, sizeof sz);
            const char *b = base_name(models[i]);
            printf("  \033[36m%-48s\033[0m \033[90m%8s\033[0m\n", b, sz);
        }
    }
    for (int i = 0; i < n; i++) free(models[i]);
    free(models);

    printf("\n\033[1mStarter picks\033[0m \033[90m(pull with /cookbook get <repo>)\033[0m\n");
    for (int i = 0; i < N_PRESETS; i++)
        printf("  \033[32m%-44s\033[0m \033[90m%s\033[0m\n",
               PRESETS[i].repo, PRESETS[i].note);

    printf("\n\033[90m  /cookbook search [query]   find trending models that fit your VRAM\n"
           "  /cookbook get <org/name>   download  ·  /cookbook rm <name>  delete\033[0m\n\n");
    fflush(stdout);
}

/* ── search ─────────────────────────────────────────────────────────── */

static void cmd_search(const char *query) {
    HwInfo hw = hw_probe();
    double vram_gb = hw.has_gpu ? hw.vram_total_mb / 1024.0 : 0.0;

    char url[1024];
    char *q = (query && *query) ? url_encode(query) : NULL;
    snprintf(url, sizeof url,
        HF_API "?sort=trendingScore&direction=-1&limit=100&filter=gguf%s%s",
        q ? "&search=" : "", q ? q : "");
    free(q);

    printf("\n\033[90m● querying HuggingFace for trending GGUF models%s%s …\033[0m\n",
           (query && *query) ? " matching " : "", (query && *query) ? query : "");
    fflush(stdout);

    char *json = http_get_json(url);
    if (!json || json[0] != '[') {
        printf("\033[31m[cookbook: HF API request failed — check your connection]\033[0m\n\n");
        free(json);
        return;
    }

    if (vram_gb > 0)
        printf("\n\033[1mTrending GGUF models\033[0m \033[90m(● fits your %s, ~%.0f GB VRAM)\033[0m\n",
               hw.gpu_name, vram_gb);
    else
        printf("\n\033[1mTrending GGUF models\033[0m \033[90m(no GPU detected — showing size estimates)\033[0m\n");

    int shown = 0;
    for (int i = 0; i < 100 && shown < 15; i++) {
        char path[32];
        snprintf(path, sizeof path, "%d.id", i);
        char *id = jx_get_string(json, path);
        if (!id) break;                        /* past end of array */
        if (!id[0]) { free(id); break; }

        snprintf(path, sizeof path, "%d.downloads", i);
        long dl = jx_get_int(json, path);

        double pb  = parse_params_b(id);
        double est = est_gb_q4(pb);
        bool fits  = (vram_gb <= 0) || (est <= 0) || (est <= vram_gb * 0.92);
        if (vram_gb > 0 && !fits) { free(id); continue; }   /* hide non-fitting */

        char estbuf[16];
        if (est > 0) snprintf(estbuf, sizeof estbuf, "~%.1fGB", est);
        else         snprintf(estbuf, sizeof estbuf, "?");

        const char *mark = fits ? "\033[32m●\033[0m" : "\033[90m○\033[0m";
        printf("  %s \033[36m%-46s\033[0m \033[90m%7s  %ld dl\033[0m\n",
               mark, id, estbuf, dl < 0 ? 0 : dl);
        shown++;
        free(id);
    }
    free(json);

    if (shown == 0)
        printf("  \033[90m(nothing matched — try a broader query, or drop the query)\033[0m\n");
    printf("\n\033[90m  → /cookbook get <org/name> to download\033[0m\n\n");
    fflush(stdout);
}

/* ── get ────────────────────────────────────────────────────────────── */

/* Collect .gguf sibling filenames from an HF model-info JSON. Returns count;
 * fills *out with a malloc'd array of malloc'd strings (caller frees). */
static int collect_gguf_siblings(const char *json, char ***out) {
    char **files = NULL;
    int count = 0, cap = 0;
    for (int i = 0; ; i++) {
        char path[40];
        snprintf(path, sizeof path, "siblings.%d.rfilename", i);
        char *fn = jx_get_string(json, path);
        if (!fn) break;
        size_t len = strlen(fn);
        if (len > 5 && strcasecmp(fn + len - 5, ".gguf") == 0) {
            if (count == cap) {
                cap = cap ? cap * 2 : 8;
                files = realloc(files, (size_t)cap * sizeof *files);
            }
            files[count++] = fn;
        } else {
            free(fn);
        }
    }
    *out = files;
    return count;
}

/* True when `name` is one shard of a multi-part GGUF (…-00001-of-00007.gguf).
 * When it is, *stem_len is set to the length of the shared prefix up to the
 * shard-number field so callers can group parts. */
static bool split_stem(const char *name, size_t *prefix_len, const char **suffix) {
    const char *of = strstr(name, "-of-");
    if (!of) return false;
    /* the 5 chars before "-of-" must be the shard index, preceded by '-' */
    if (of - name < 6) return false;
    for (const char *d = of - 5; d < of; d++)
        if (!isdigit((unsigned char)*d)) return false;
    if (of[-6] != '-') return false;
    /* suffix "-of-#####.gguf" must be all digits then .gguf */
    const char *s = of + 4;
    int digits = 0;
    while (isdigit((unsigned char)*s)) { s++; digits++; }
    if (digits == 0 || strcasecmp(s, ".gguf") != 0) return false;
    *prefix_len = (size_t)(of - 5 - name);   /* up to (not incl.) shard index */
    *suffix = of;                            /* "-of-#####.gguf" */
    return true;
}

/* Download one file via curl (foreground, resumable, live progress bar).
 * Returns true on a verified-complete local file. */
static bool download_one(const char *repo, const char *rfilename, const char *dstdir) {
    if (!safe_rfilename(rfilename)) {
        printf("\033[31m[cookbook: refusing unsafe filename '%s']\033[0m\n", rfilename);
        return false;
    }
    char url[1200];
    snprintf(url, sizeof url,
        "https://huggingface.co/%s/resolve/main/%s?download=true", repo, rfilename);

    const char *base = base_name(rfilename);
    char dest[PATH_MAX];
    snprintf(dest, sizeof dest, "%s/%s", dstdir, base);

    long rsize = remote_size(url);
    char szbuf[24]; human_size(rsize, szbuf, sizeof szbuf);

    struct stat st;
    if (stat(dest, &st) == 0 && rsize > 0 && st.st_size == rsize) {
        printf("  \033[32m✓\033[0m %s \033[90m(already downloaded, %s)\033[0m\n", base, szbuf);
        return true;
    }

    printf("  \033[36m↓\033[0m %s \033[90m(%s)\033[0m\n", base, szbuf);
    fflush(stdout);

    char cmd[PATH_MAX + 1600];
    snprintf(cmd, sizeof cmd,
        "curl -L --fail --progress-bar --retry 3 -C - -A '" COOKBOOK_UA "' "
        "-o '%s' '%s'", dest, url);
    int rc = system(cmd);

    /* curl exits non-zero on a range-not-satisfiable (file already complete);
     * treat "file exists at full size" as success regardless of rc. */
    if (stat(dest, &st) == 0 && (rsize <= 0 ? st.st_size > 0 : st.st_size == rsize))
        return true;

    printf("\033[31m[cookbook: download of %s failed (curl exit %d)]\033[0m\n",
           base, rc == -1 ? -1 : (rc >> 8));
    return false;
}

static void cmd_get(const char *rest) {
    char repo[256] = "", quant[64] = "";
    /* first token = repo, optional second = quant hint */
    if (sscanf(rest, "%255s %63s", repo, quant) < 1 || !repo[0]) {
        printf("\033[31m[/cookbook get: usage: /cookbook get <org/name> [quant]]\033[0m\n");
        return;
    }
    if (!valid_repo(repo)) {
        printf("\033[31m[/cookbook get: '%s' is not a valid 'org/name' repo id]\033[0m\n", repo);
        return;
    }

    char apiurl[512];
    snprintf(apiurl, sizeof apiurl, HF_API "/%s", repo);
    printf("\n\033[90m● resolving %s …\033[0m\n", repo);
    fflush(stdout);

    char *json = http_get_json(apiurl);
    if (!json) {
        printf("\033[31m[/cookbook get: network error reaching HuggingFace]\033[0m\n\n");
        return;
    }
    char *apierr = jx_get_string(json, "error");
    if (apierr) {
        printf("\033[31m[/cookbook get: HF says: %s]\033[0m\n\n", apierr);
        free(apierr); free(json);
        return;
    }

    char **files = NULL;
    int nf = collect_gguf_siblings(json, &files);
    free(json);
    if (nf == 0) {
        printf("\033[31m[/cookbook get: no .gguf files in %s — is it a GGUF repo?]\033[0m\n"
               "\033[90m  (BASI runs GGUF only; safetensors repos need conversion first)\033[0m\n\n",
               repo);
        return;
    }

    /* pick the file */
    int chosen = -1;
    if (quant[0]) {
        for (int i = 0; i < nf; i++)
            if (ci_find(files[i], quant)) { chosen = i; break; }
        if (chosen < 0) {
            printf("\033[31m[/cookbook get: no GGUF matching '%s'. Available:]\033[0m\n", quant);
            for (int i = 0; i < nf; i++) printf("    %s\n", base_name(files[i]));
            printf("\n");
            for (int i = 0; i < nf; i++) free(files[i]);
            free(files);
            return;
        }
    } else {
        for (int p = 0; QUANT_PREFS[p] && chosen < 0; p++)
            for (int i = 0; i < nf; i++)
                if (ci_find(files[i], QUANT_PREFS[p])) { chosen = i; break; }
        if (chosen < 0) chosen = 0;   /* nothing recognized — take the first */
    }

    char dstdir[PATH_MAX];
    models_dir(dstdir, sizeof dstdir);
    if (mkdir_p(dstdir) != 0) {
        printf("\033[31m[/cookbook get: cannot create %s (%s)]\033[0m\n\n",
               dstdir, strerror(errno));
        for (int i = 0; i < nf; i++) free(files[i]);
        free(files);
        return;
    }

    /* multi-part? gather every shard that shares the chosen file's stem. */
    size_t plen; const char *suf;
    int ok = 0, attempted = 0;
    if (split_stem(files[chosen], &plen, &suf)) {
        printf("\033[90m  multi-part model → fetching all shards into %s\033[0m\n", dstdir);
        for (int i = 0; i < nf; i++) {
            size_t ip; const char *is;
            if (split_stem(files[i], &ip, &is) && ip == plen && strcmp(is, suf) == 0 &&
                strncmp(files[i], files[chosen], plen) == 0) {
                attempted++;
                if (download_one(repo, files[i], dstdir)) ok++;
            }
        }
    } else {
        printf("\033[90m  → %s\033[0m\n", dstdir);
        attempted = 1;
        if (download_one(repo, files[chosen], dstdir)) ok++;
    }

    char tagbuf[128];
    snprintf(tagbuf, sizeof tagbuf, "%s", base_name(files[chosen]));
    size_t tl = strlen(tagbuf);            /* strip .gguf for the /model hint */
    if (tl > 5 && strcasecmp(tagbuf + tl - 5, ".gguf") == 0) tagbuf[tl - 5] = 0;

    for (int i = 0; i < nf; i++) free(files[i]);
    free(files);

    if (ok == attempted && ok > 0) {
        printf("\n\033[32m✓ downloaded.\033[0m Switch with \033[36m/model %s\033[0m\n\n", tagbuf);
    } else {
        printf("\n\033[31m[/cookbook get: %d/%d parts failed]\033[0m\n\n",
               attempted - ok, attempted);
    }
    fflush(stdout);
}

/* ── rm ─────────────────────────────────────────────────────────────── */

static void cmd_rm(const char *arg) {
    while (*arg == ' ') arg++;
    if (!*arg) {
        printf("\033[31m[/cookbook rm: usage: /cookbook rm <name-substring>]\033[0m\n");
        return;
    }
    char **models = NULL;
    int n = basi_list_models(&models);
    int match = -1, matches = 0;
    for (int i = 0; i < n; i++)
        if (ci_find(base_name(models[i]), arg)) { match = i; matches++; }

    if (matches == 0) {
        printf("\033[31m[/cookbook rm: no cached model matches '%s']\033[0m\n", arg);
    } else if (matches > 1) {
        printf("\033[33m[/cookbook rm: '%s' matches %d models — be more specific:]\033[0m\n",
               arg, matches);
        for (int i = 0; i < n; i++)
            if (ci_find(base_name(models[i]), arg)) printf("    %s\n", base_name(models[i]));
    } else {
        const char *path = models[match];
        /* If this file lives inside an HF hub cache (…/models--org--name/…),
         * removing just the .gguf leaves the blob behind — offer the whole
         * repo dir instead. Otherwise it's a plain file we downloaded. */
        char target[PATH_MAX];
        bool is_dir = false;
        const char *hub = strstr(path, "/models--");
        if (hub) {
            const char *slash = strchr(hub + 1, '/');   /* end of models--… segment */
            size_t tlen = slash ? (size_t)(slash - path) : strlen(path);
            if (tlen < sizeof target) {
                memcpy(target, path, tlen); target[tlen] = 0;
                is_dir = true;
            } else {
                snprintf(target, sizeof target, "%s", path);
            }
        } else {
            snprintf(target, sizeof target, "%s", path);
        }

        printf("\033[33mDelete %s\033[0m\n  %s\n\033[33mThis frees the disk but is not reversible. Proceed? [y/N] \033[0m",
               is_dir ? "this cached repo" : "this model file", target);
        fflush(stdout);
        char resp[16] = {0};
        if (fgets(resp, sizeof resp, stdin) && (resp[0] == 'y' || resp[0] == 'Y')) {
            int rc;
            if (is_dir) {
                char cmd[PATH_MAX + 32];
                snprintf(cmd, sizeof cmd, "rm -rf '%s'", target);
                rc = system(cmd);
                rc = (rc == -1) ? -1 : (rc >> 8);
            } else {
                rc = unlink(target);
            }
            if (rc == 0) printf("\033[90m[deleted %s]\033[0m\n", target);
            else printf("\033[31m[/cookbook rm: delete failed (%s)]\033[0m\n", strerror(errno));
        } else {
            printf("\033[90m[cancelled]\033[0m\n");
        }
    }
    for (int i = 0; i < n; i++) free(models[i]);
    free(models);
    fflush(stdout);
}

/* ── help ───────────────────────────────────────────────────────────── */

static void cmd_help(void) {
    printf(
        "\n\033[1m/cookbook\033[0m — download and manage local GGUF models\n"
        "  \033[36m/cookbook\033[0m                    list cached models + starter picks\n"
        "  \033[36m/cookbook search\033[0m [query]     trending GGUF models that fit your VRAM\n"
        "  \033[36m/cookbook get\033[0m <org/name> [q] download a GGUF (q = quant hint, e.g. Q5_K_M)\n"
        "  \033[36m/cookbook rm\033[0m <name>          delete a cached model\n"
        "\n\033[90m  After a get, switch to it with /model <name>. Files land in ~/models.\033[0m\n\n");
    fflush(stdout);
}

/* ── dispatch ───────────────────────────────────────────────────────── */

void cookbook_command(const char *args) {
    const char *p = args ? args : "";
    while (*p == ' ') p++;

    char sub[32] = "";
    int i = 0;
    while (p[i] && p[i] != ' ' && i < (int)sizeof(sub) - 1) { sub[i] = p[i]; i++; }
    sub[i] = 0;
    const char *rest = p + i;
    while (*rest == ' ') rest++;

    if (!sub[0] || !strcmp(sub, "list") || !strcmp(sub, "ls"))
        cmd_list();
    else if (!strcmp(sub, "search") || !strcmp(sub, "find"))
        cmd_search(rest);
    else if (!strcmp(sub, "get") || !strcmp(sub, "download") ||
             !strcmp(sub, "pull") || !strcmp(sub, "add"))
        cmd_get(rest);
    else if (!strcmp(sub, "rm") || !strcmp(sub, "remove") || !strcmp(sub, "delete"))
        cmd_rm(rest);
    else if (!strcmp(sub, "help") || !strcmp(sub, "-h") || !strcmp(sub, "--help"))
        cmd_help();
    else if (strchr(sub, '/'))          /* "/cookbook org/name" → get shorthand */
        cmd_get(p);
    else {
        printf("\033[31m[/cookbook: unknown subcommand '%s']\033[0m\n", sub);
        cmd_help();
    }
}
