/* Server-backed generation (M1 spike). See srvgen.h. */
#include "srvgen.h"
#include "util.h"      /* jx_get_string */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* JSON-escape a string (no surrounding quotes). Caller frees. */
static char *json_escape(const char *s) {
    size_t n = strlen(s), cap = n * 2 + 16, k = 0;
    char *o = malloc(cap);
    if (!o) return NULL;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char) s[i];
        if (k + 8 > cap) { cap *= 2; char *t = realloc(o, cap); if (!t) { free(o); return NULL; } o = t; }
        switch (c) {
            case '"':  o[k++] = '\\'; o[k++] = '"';  break;
            case '\\': o[k++] = '\\'; o[k++] = '\\'; break;
            case '\n': o[k++] = '\\'; o[k++] = 'n';  break;
            case '\r': o[k++] = '\\'; o[k++] = 'r';  break;
            case '\t': o[k++] = '\\'; o[k++] = 't';  break;
            default:
                if (c < 0x20) k += (size_t) snprintf(o + k, 8, "\\u%04x", c);
                else o[k++] = (char) c;
        }
    }
    o[k] = 0;
    return o;
}

pid_t srvgen_spawn(const char *server_bin, const char *model_path, int ngl, int ctx,
                   const char *extra, int port, const char *logpath, int timeout_s) {
    char nglbuf[16], ctxbuf[16], portbuf[16];
    snprintf(nglbuf,  sizeof nglbuf,  "%d", ngl);
    snprintf(ctxbuf,  sizeof ctxbuf,  "%d", ctx);
    snprintf(portbuf, sizeof portbuf, "%d", port);

    /* base argv + up to 24 tokens from `extra` (space-split) */
    char *argv[48];
    int a = 0;
    argv[a++] = (char *) server_bin;
    argv[a++] = "-m";   argv[a++] = (char *) model_path;
    argv[a++] = "-ngl"; argv[a++] = nglbuf;
    argv[a++] = "-c";   argv[a++] = ctxbuf;
    argv[a++] = "--host"; argv[a++] = "127.0.0.1";
    argv[a++] = "--port"; argv[a++] = portbuf;

    char *extra_copy = extra ? strdup(extra) : NULL;
    if (extra_copy) {
        char *save = NULL, *tok = strtok_r(extra_copy, " ", &save);
        while (tok && a < 46) { argv[a++] = tok; tok = strtok_r(NULL, " ", &save); }
    }
    argv[a] = NULL;

    pid_t pid = fork();
    if (pid < 0) { free(extra_copy); return -1; }
    if (pid == 0) {
        /* child: redirect stdout+stderr to the log, then exec the server */
        int fd = open(logpath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) { dup2(fd, 1); dup2(fd, 2); close(fd); }
        setsid();  /* own process group so srvgen_kill can't miss it */
        execvp(server_bin, argv);
        _exit(127);
    }
    free(extra_copy);

    /* parent: poll /health until the server is up */
    char cmd[256];
    snprintf(cmd, sizeof cmd,
             "curl -sf -o /dev/null http://127.0.0.1:%d/health", port);
    for (int i = 0; i < timeout_s; i++) {
        /* if the child died, bail */
        int st; if (waitpid(pid, &st, WNOHANG) == pid) return -1;
        if (system(cmd) == 0) return pid;   /* ready */
        sleep(1);
    }
    srvgen_kill(pid);
    return -1;
}

char *srvgen_complete(int port, const char *prompt, int n_predict, double temp,
                      int greedy, const char *grammar,
                      void (*emit)(const char *chunk, void *ud), void *ud,
                      double *tps, int *n_out) {
    char *esc = json_escape(prompt);
    if (!esc) return NULL;
    char *gesc = grammar && *grammar ? json_escape(grammar) : NULL;

    /* write the request body to a temp file (avoids quoting a big prompt on argv) */
    char reqpath[] = "/tmp/basi_srvreq_XXXXXX";
    int rfd = mkstemp(reqpath);
    if (rfd < 0) { free(esc); free(gesc); return NULL; }
    FILE *rf = fdopen(rfd, "w");
    fprintf(rf, "{\"prompt\":\"%s\",\"n_predict\":%d,\"temperature\":%.3f,%s",
            esc, n_predict, temp, greedy ? "\"top_k\":1," : "");
    if (gesc) fprintf(rf, "\"grammar\":\"%s\",", gesc);
    fprintf(rf, "\"cache_prompt\":true,\"stream\":true}");
    fclose(rf);
    free(esc); free(gesc);

    char cmd[512];
    snprintf(cmd, sizeof cmd,
        "curl -N -s -X POST http://127.0.0.1:%d/completion "
        "-H 'Content-Type: application/json' --data-binary @%s", port, reqpath);

    FILE *p = popen(cmd, "r");
    if (!p) { unlink(reqpath); return NULL; }

    StringBuf out; sb_init(&out);
    char *line = NULL; size_t cap = 0; ssize_t len;
    int n_tok = 0; int done = 0;
    double t0 = now_s();
    while ((len = getline(&line, &cap, p)) != -1) {
        /* SSE frames: "data: {json}\n" */
        char *j = strstr(line, "data:");
        if (!j) continue;
        j += 5; while (*j == ' ') j++;
        char *content = jx_get_string(j, "content");
        if (content) {
            if (content[0]) { sb_append_str(&out, content); if (emit) emit(content, ud); n_tok++; }
            free(content);
        }
        if (strstr(j, "\"stop\":true")) { done = 1; break; }
    }
    free(line);
    pclose(p);
    unlink(reqpath);

    double dt = now_s() - t0;
    if (tps)   *tps = (dt > 0) ? n_tok / dt : 0.0;
    if (n_out) *n_out = n_tok;
    (void) done;
    return sb_to_str(&out);
}

void srvgen_kill(pid_t pid) {
    if (pid <= 0) return;
    kill(-pid, SIGTERM);       /* whole process group (setsid in child) */
    for (int i = 0; i < 30; i++) {
        int st; if (waitpid(pid, &st, WNOHANG) == pid) return;
        usleep(100000);
    }
    kill(-pid, SIGKILL);
    waitpid(pid, NULL, 0);
}

static void print_emit(const char *chunk, void *ud) {
    (void) ud;
    fputs(chunk, stdout); fflush(stdout);
}

void srvgen_selftest(const char *model_path, int ngl, int ctx) {
    const char *server_bin = getenv("BASI_SERVER_BIN");
    if (!server_bin || !*server_bin)
        server_bin = "/home/alberto/llama.cpp/build_vulkan/bin/llama-server";
    const int port = 8181;

    /* spec flags from env: BASI_SPEC=draft-mtp BASI_SPEC_NMAX=1 -> MTP for free */
    char extra[128] = "";
    const char *spec = getenv("BASI_SPEC");
    if (spec && *spec) {
        int nmax = 1; const char *e = getenv("BASI_SPEC_NMAX"); if (e && *e) nmax = atoi(e);
        snprintf(extra, sizeof extra, "-fa on --spec-type %s --spec-draft-n-max %d", spec, nmax);
    }

    int n_predict = 128; { const char *e = getenv("BASI_SPEC_N"); if (e && *e) n_predict = atoi(e); }
    const char *prompt = getenv("BASI_SPEC_PROMPT");
    if (!prompt || !*prompt)
        prompt = "Write a C function that reverses a string in place. Explain your approach first.";

    fprintf(stderr, "\n=== server-gen self-test: ngl=%d ctx=%d spec='%s' ===\n", ngl, ctx, extra);
    fprintf(stderr, "[srv] spawning llama-server (loads the model — the ONLY copy)...\n");
    double ts0 = now_s();
    pid_t pid = srvgen_spawn(server_bin, model_path, ngl, ctx, extra[0] ? extra : NULL,
                             port, "/tmp/basi_srvgen.log", 300);
    if (pid < 0) { fprintf(stderr, "[srv] spawn/health FAILED (see /tmp/basi_srvgen.log)\n"); return; }
    fprintf(stderr, "[srv] ready in %.1fs. streaming completion:\n---\n", now_s() - ts0);

    /* optional grammar round-trip test: BASI_SRV_GRAMMAR="root ::= \"yes\" | \"no\"" */
    const char *grammar = getenv("BASI_SRV_GRAMMAR");
    if (grammar && *grammar) fprintf(stderr, "[srv] sending grammar (%zu bytes)\n", strlen(grammar));

    double tps = 0; int n = 0;
    char *txt = srvgen_complete(port, prompt, n_predict, 0.0, 1, grammar, print_emit, NULL, &tps, &n);
    fprintf(stderr, "\n---\n[srv] %d tokens, %.2f tok/s\n", n, tps);

    free(txt);
    srvgen_kill(pid);
    fprintf(stderr, "[srv] server killed. done.\n");
}
