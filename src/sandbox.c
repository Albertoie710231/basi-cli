#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sandbox.h"

/* The agent's tools run as the user, so without this a model can change any file
 * the user can. Measured 2026-09-24: Qwen3.6-27B, asked to add a slash command in
 * a scratch copy, read its working directory, walked up to the real BASI-CLI
 * repo and edited src/main.c there. The model is the part we cannot trust.
 *
 * Same model as Claude Code's sandbox: WRITES only where the work is (the project,
 * /tmp, ~/.cache, BASI's own config and data); READS everywhere except a deny
 * list of secrets, which are hidden. BASI re-executes itself under bubblewrap at
 * startup, so every tool, shell command and subprocess (llama-server, MCP
 * servers, SearXNG) inherits the limits — one place to get right. */

#define MAX_ARGS 512

static const char *const WRITABLE_HOME[] = {
    ".cache",                  /* model downloads, npm/pip caches MCP servers need */
    ".config/basi-cli",        /* saved backend/model, telemetry choice */
    ".local/share/basi-cli",   /* sessions, telemetry */
    NULL
};

/* Hidden entirely: keys, tokens, other agents' memory, browser profiles. */
static const char *const DENY_HOME[] = {
    ".ssh", ".gnupg", ".claude", ".claude.json", ".aws", ".azure",
    ".config/gcloud", ".kube", ".docker", ".netrc", ".password-store",
    ".local/share/keyrings", ".config/gh", ".git-credentials", ".npmrc",
    ".pypirc", ".mozilla", ".config/chromium", ".config/google-chrome",
    ".config/BraveSoftware", NULL
};

static bool exists(const char *p, bool *is_dir) {
    struct stat st;
    if (stat(p, &st) != 0) return false;
    if (is_dir) *is_dir = S_ISDIR(st.st_mode);
    return true;
}

static const char *find_bwrap(void) {
    static char buf[PATH_MAX];
    const char *path = getenv("PATH");
    if (!path) path = "/usr/bin:/bin";
    const char *p = path;
    while (*p) {
        const char *c = strchr(p, ':');
        size_t n = c ? (size_t)(c - p) : strlen(p);
        if (n && n < sizeof buf - 7) {
            snprintf(buf, sizeof buf, "%.*s/bwrap", (int)n, p);
            if (access(buf, X_OK) == 0) return buf;
        }
        if (!c) break;
        p = c + 1;
    }
    return NULL;
}

bool sandbox_disabled_by_args(int argc, char **argv) {
    const char *e = getenv("BASI_SANDBOX");
    if (e && (strcmp(e, "0") == 0 || strcmp(e, "off") == 0)) return true;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--no-sandbox") == 0) return true;
    return false;
}

/* argv storage: every added string is strdup'd so the list owns it */
static char *args[MAX_ARGS];
static int   nargs = 0;
static void add(const char *s) { if (nargs < MAX_ARGS - 1) args[nargs++] = strdup(s); }
static void add3(const char *a, const char *b, const char *c) { add(a); add(b); add(c); }

static void expand(const char *in, const char *home, char *out, size_t n) {
    if (in[0] == '~' && (in[1] == '/' || in[1] == '\0')) snprintf(out, n, "%s%s", home, in + 1);
    else if (in[0] == '/') snprintf(out, n, "%s", in);
    else {                                           /* relative to the project */
        char cwd[PATH_MAX];
        if (!getcwd(cwd, sizeof cwd)) strcpy(cwd, ".");
        size_t lc = strlen(cwd), li = strlen(in);
        if (lc + 1 + li + 1 > n) { out[0] = '\0'; return; }   /* too long: skipped */
        memcpy(out, cwd, lc);
        out[lc] = '/';
        memcpy(out + lc + 1, in, li + 1);
    }
}

/* One allow entry: "<path> [rw|ro]" — default rw (the reason to list a path is
 * usually to let the agent work there). */
static void add_allow(const char *spec, const char *home, int *count) {
    char path[PATH_MAX], mode[8] = "rw";
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof tmp, "%s", spec);
    char *sp = strpbrk(tmp, " \t");
    if (sp) {
        *sp = '\0';
        char *m = sp + 1;
        while (*m == ' ' || *m == '\t') m++;
        if (strncmp(m, "ro", 2) == 0) strcpy(mode, "ro");
    }
    expand(tmp, home, path, sizeof path);
    if (!exists(path, NULL)) {
        fprintf(stderr, "\033[33m[sandbox] allow-list path not found, skipped: %s\033[0m\n", path);
        return;
    }
    if (strcmp(mode, "rw") == 0) { add3("--bind", path, path); (*count)++; }
    /* "ro" needs nothing: everything is readable already */
}

void sandbox_maybe_reexec(int argc, char **argv) {
    if (getenv("BASI_SANDBOXED")) return;                 /* already inside */
    if (sandbox_disabled_by_args(argc, argv)) {
        fprintf(stderr, "\033[33m[sandbox] OFF — the agent can change any file you can.\033[0m\n");
        return;
    }
    const char *bwrap = find_bwrap();
    if (!bwrap) {
        fprintf(stderr, "\033[33m[sandbox] bubblewrap (bwrap) not found — running WITHOUT a sandbox; "
                        "the agent can change any file you can. Install bubblewrap to enable it.\033[0m\n");
        return;
    }
    const char *home = getenv("HOME");
    char self[PATH_MAX], cwd[PATH_MAX];
    ssize_t sl = readlink("/proc/self/exe", self, sizeof self - 1);
    if (sl <= 0 || !getcwd(cwd, sizeof cwd) || !home) return;
    self[sl] = '\0';

    add(bwrap);
    add3("--ro-bind", "/", "/");                 /* read everything ... */
    add3("--dev-bind", "/dev", "/dev");          /* GPUs, ptys */
    add("--proc"); add("/proc");
    add3("--bind", "/tmp", "/tmp");              /* ... write here, */
    add3("--bind", cwd, cwd);                    /* in the project, */
    char p[PATH_MAX];
    for (int i = 0; WRITABLE_HOME[i]; i++) {     /* and BASI's own places */
        snprintf(p, sizeof p, "%s/%s", home, WRITABLE_HOME[i]);
        if (exists(p, NULL)) add3("--bind", p, p);
    }
    int allowed = 0;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--allow") == 0 && i + 1 < argc) add_allow(argv[++i], home, &allowed);
    FILE *f = fopen(".basi/sandbox", "r");
    if (f) {
        char line[PATH_MAX + 16];
        while (fgets(line, sizeof line, f)) {
            line[strcspn(line, "\r\n")] = '\0';
            char *s = line;
            while (*s == ' ' || *s == '\t') s++;
            if (*s && *s != '#') add_allow(s, home, &allowed);
        }
        fclose(f);
    }
    /* Deny list last, so it wins even inside a writable or allowed path. */
    int hidden = 0;
    for (int i = 0; DENY_HOME[i]; i++) {
        bool dir = false;
        snprintf(p, sizeof p, "%s/%s", home, DENY_HOME[i]);
        if (!exists(p, &dir)) continue;
        if (dir) { add("--tmpfs"); add(p); }
        else     add3("--ro-bind", "/dev/null", p);
        hidden++;
    }
    add3("--setenv", "BASI_SANDBOXED", "1");
    add("--");
    add(self);
    for (int i = 1; i < argc; i++) add(argv[i]);
    args[nargs] = NULL;

    bool home_is_cwd = strcmp(cwd, home) == 0;
    fprintf(stderr, "\033[90m[sandbox] writes: this project, /tmp, ~/.cache, BASI's data%s; "
                    "%d secret path(s) hidden (~/.ssh, ~/.claude, …). "
                    "--allow <dir> / .basi/sandbox to grant more, --no-sandbox to turn off.\033[0m\n",
            allowed ? " + allow-list" : "", hidden);
    if (home_is_cwd)
        fprintf(stderr, "\033[33m[sandbox] running from your home directory: all of ~ is writable.\033[0m\n");
    if (getenv("BASI_SANDBOX_DEBUG")) {
        for (int i = 0; i < nargs; i++) fprintf(stderr, "%s%s", i ? " " : "[sandbox] ", args[i]);
        fprintf(stderr, "\n");
    }
    fflush(stderr);
    execv(bwrap, args);
    /* exec failed (e.g. user namespaces disabled): say so and continue unsandboxed */
    fprintf(stderr, "\033[33m[sandbox] could not start bwrap (%s) — running WITHOUT a sandbox.\033[0m\n",
            strerror(errno));
}
