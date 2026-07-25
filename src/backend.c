/* backend.c — the list of selectable llama-server binaries. See backend.h. */
#include "backend.h"
#include "util.h"      /* run_command_status */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>    /* access — is the binary actually there and runnable? */

static Backend g_list[BACKEND_MAX];
static int     g_n      = 0;
static int     g_loaded = 0;
static char    g_sel[32] = "";   /* backend_select()'s choice */

/* Scratch entry for the $BASI_SERVER_BIN override, which names a binary that
 * need not appear in the config at all. */
static Backend g_env_backend;

void backend_config_path(char *out, size_t n) {
    const char *xdg  = getenv("XDG_CONFIG_HOME");
    const char *home = getenv("HOME");
    if (xdg && *xdg)        snprintf(out, n, "%s/basi-cli/backends", xdg);
    else if (home && *home) snprintf(out, n, "%s/.config/basi-cli/backends", home);
    else                    snprintf(out, n, ".basi/backends");
}

/* Trim ASCII whitespace in place; returns the first non-space byte. */
static char *trim(char *s) {
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
    size_t l = strlen(s);
    while (l > 0 && (s[l-1] == ' ' || s[l-1] == '\t' ||
                     s[l-1] == '\r' || s[l-1] == '\n')) s[--l] = '\0';
    return s;
}

/* Copy `src` into `dst`, expanding a leading "~/" against $HOME. Config files are
 * hand-written, so a tilde path is likely and failing on it would be unhelpful. */
static void copy_path(char *dst, size_t cap, const char *src) {
    const char *home = getenv("HOME");
    if (src[0] == '~' && src[1] == '/' && home && *home)
        snprintf(dst, cap, "%s%s", home, src + 1);
    else
        snprintf(dst, cap, "%s", src);
}

/* The built-in fallback, used when nothing is declared. */
static void seed_default(void) {
    g_n = 1;
    snprintf(g_list[0].name, sizeof g_list[0].name, "%s", "vulkan");
    copy_path(g_list[0].server_bin, sizeof g_list[0].server_bin, BASI_DEFAULT_SERVER_BIN);
    g_list[0].extra_flags[0] = '\0';
}

int backend_load(void) {
    if (g_loaded) return g_n;
    g_loaded = 1;
    g_n = 0;

    char path[600];
    backend_config_path(path, sizeof path);
    FILE *f = fopen(path, "r");
    if (!f) { seed_default(); return g_n; }

    /* Format, one per line:  name = <path to llama-server> [| default flags] */
    char line[1024];
    while (fgets(line, sizeof line, f) && g_n < BACKEND_MAX) {
        char *s = trim(line);
        if (!*s || *s == '#') continue;

        char *eq = strchr(s, '=');
        if (!eq) continue;                    /* not a declaration; skip quietly */
        *eq = '\0';
        char *name = trim(s);
        char *rest = trim(eq + 1);
        if (!*name || !*rest) continue;

        /* Optional per-binary default flags after a '|'. */
        char *flags = strchr(rest, '|');
        if (flags) { *flags = '\0'; flags = trim(flags + 1); rest = trim(rest); }

        Backend *b = &g_list[g_n];
        snprintf(b->name, sizeof b->name, "%s", name);
        copy_path(b->server_bin, sizeof b->server_bin, rest);
        snprintf(b->extra_flags, sizeof b->extra_flags, "%s", flags ? flags : "");
        g_n++;
    }
    fclose(f);

    if (g_n == 0) seed_default();             /* empty or all-comments file */
    return g_n;
}

const Backend *backend_list(int *n) {
    backend_load();
    if (n) *n = g_n;
    return g_list;
}

const Backend *backend_get(const char *name) {
    if (!name || !*name) return NULL;
    backend_load();
    for (int i = 0; i < g_n; i++)
        if (strcmp(g_list[i].name, name) == 0) return &g_list[i];
    return NULL;
}

int backend_select(const char *name) {
    if (!backend_get(name)) return 0;
    snprintf(g_sel, sizeof g_sel, "%s", name);
    return 1;
}

const char *backend_selected_name(void) { return g_sel; }

const Backend *backend_active(void) {
    backend_load();

    /* 1. $BASI_SERVER_BIN — predates this module; keep it absolute so existing
     *    harnesses (and the factory) behave exactly as before. */
    const char *env_bin = getenv("BASI_SERVER_BIN");
    if (env_bin && *env_bin) {
        snprintf(g_env_backend.name, sizeof g_env_backend.name, "%s", "env");
        copy_path(g_env_backend.server_bin, sizeof g_env_backend.server_bin, env_bin);
        g_env_backend.extra_flags[0] = '\0';
        return &g_env_backend;
    }

    /* 2. $BASI_BACKEND by name. */
    const Backend *b = backend_get(getenv("BASI_BACKEND"));
    if (b) return b;

    /* 3. The picker's choice / the saved `backend=` line. */
    if (g_sel[0] && (b = backend_get(g_sel)) != NULL) return b;

    /* 4. Vulkan runs anywhere, so it is the compatibility default. */
    if ((b = backend_get("vulkan")) != NULL) return b;
    return &g_list[0];
}

const char *backend_probe_hint(const Backend *b, const char *err) {
    /* Ask the filesystem, don't parse the error text: the shell localizes its
     * messages ("No existe el fichero" on a Spanish system), so string-matching
     * English would silently drop the hint exactly when it is needed. */
    if (b && b->server_bin[0] && access(b->server_bin, X_OK) != 0)
        return "fix the path in the backends config";

    if (!err || !*err) return "";
    /* A SYCL build links the Intel compiler runtime, which lives only under the
     * oneAPI prefix and is not registered with the system loader — so the give-away
     * is one of these names, and the fix is the env script. These come from ld.so,
     * which is not localized, so matching them is safe. */
    static const char *intel_libs[] = { "libsvml", "libimf", "libintlc", "libirng",
                                        "libsycl", "libur_", "libOpenCL" };
    for (size_t i = 0; i < sizeof intel_libs / sizeof *intel_libs; i++) {
        if (strstr(err, intel_libs[i])) {
            static const char *setvars = "/opt/intel/oneapi/setvars.sh";
            if (access(setvars, R_OK) == 0) return "source /opt/intel/oneapi/setvars.sh";
            return "put the oneAPI runtime libraries on LD_LIBRARY_PATH";
        }
    }
    return "";
}

int backend_probe(const Backend *b, char *err, size_t errn) {
    if (err && errn) err[0] = '\0';
    if (!b || !b->server_bin[0]) {
        if (err) snprintf(err, errn, "no llama-server binary configured");
        return -1;
    }

    /* A missing or non-executable path is answerable without running anything, and
     * gives a cleaner message than the shell's localized "not found". */
    if (access(b->server_bin, X_OK) != 0) {
        if (err) snprintf(err, errn, "%s: not found or not executable", b->server_bin);
        return -1;
    }

    /* `--version` is the cheapest real load test: the dynamic loader has to
     * resolve every dependency before main() runs, so a missing runtime env fails
     * here exactly as it would at launch — but with the message in hand instead of
     * buried in /tmp/basi_srvgen.log. 2>&1 because the loader writes to stderr. */
    char cmd[700];
    snprintf(cmd, sizeof cmd, "\"%s\" --version 2>&1", b->server_bin);

    int code = -1;
    char *out = run_command_status(cmd, 8 * 1024, &code);
    if (code == 0) { free(out); return 0; }

    /* Report the loader's own words — "libsvml.so: cannot open shared object
     * file" is the actionable part, and paraphrasing it would lose that. */
    if (err && errn) {
        char first[400] = "";
        if (out) {
            snprintf(first, sizeof first, "%s", out);
            char *nl = strchr(first, '\n');
            if (nl) *nl = '\0';
        }
        if (first[0]) snprintf(err, errn, "%s", trim(first));
        else          snprintf(err, errn, "%s did not run (exit %d)", b->server_bin, code);
    }
    free(out);
    return -1;
}
