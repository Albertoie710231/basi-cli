#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <errno.h>
#include <unistd.h>
#include <signal.h>
#include <poll.h>
#include <dirent.h>

#include "util.h"

/* ── StringBuf ─────────────────────────────────────────────────────── */

void sb_init(StringBuf *sb) {
    sb->data = NULL;
    sb->len  = 0;
    sb->cap  = 0;
}

void sb_free(StringBuf *sb) {
    free(sb->data);
    sb->data = NULL;
    sb->len = sb->cap = 0;
}

void sb_ensure(StringBuf *sb, size_t extra) {
    size_t need = sb->len + extra;
    if (need <= sb->cap) return;
    size_t newcap = sb->cap ? sb->cap * 2 : 256;
    while (newcap < need) newcap *= 2;
    /* Save into a temp so a failed realloc doesn't leak the original block
     * (CERT MEM32-C / PVS V701); StringBuf is infrastructure, so abort. */
    char *tmp = realloc(sb->data, newcap);
    if (!tmp) { perror("realloc"); abort(); }
    sb->data = tmp;
    sb->cap = newcap;
}

void sb_append(StringBuf *sb, const char *s, size_t n) {
    sb_ensure(sb, n);
    memcpy(sb->data + sb->len, s, n);
    sb->len += n;
}

void sb_append_str(StringBuf *sb, const char *s) {
    sb_append(sb, s, strlen(s));
}

void sb_append_char(StringBuf *sb, char c) {
    sb_ensure(sb, 1);
    sb->data[sb->len++] = c;
}

char *sb_to_str(StringBuf *sb) {
    sb_ensure(sb, 1);
    sb->data[sb->len] = '\0';
    return sb->data;
}

void sb_clear(StringBuf *sb) {
    sb->len = 0;
}

/* ── Tokenize command string (respects quotes) ─────────────────────── */

ArgList tokenize_command(const char *cmd) {
    ArgList al = { NULL, 0 };
    int cap = 0;
    const char *p = cmd;

    while (*p) {
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;

        const char *start;
        const char *end;

        if (*p == '"' || *p == '\'') {
            char quote = *p++;
            start = p;
            while (*p && *p != quote) p++;
            end = p;
            if (*p) p++;
        } else {
            start = p;
            while (*p && *p != ' ' && *p != '\t') p++;
            end = p;
        }

        if (end > start) {
            if (al.count >= cap) {
                cap = cap ? cap * 2 : 8;
                char **tmp = realloc(al.args, cap * sizeof(char *));
                if (!tmp) { perror("realloc"); abort(); }
                al.args = tmp;
            }
            size_t len = end - start;
            al.args[al.count] = malloc(len + 1);
            memcpy(al.args[al.count], start, len);
            al.args[al.count][len] = '\0';
            al.count++;
        }
    }
    return al;
}

void arglist_free(ArgList *al) {
    for (int i = 0; i < al->count; i++)
        free(al->args[i]);
    free(al->args);
    al->args = NULL;
    al->count = 0;
}

/* ── Run a shell command and capture output ────────────────────────── */

char *run_command_status(const char *cmd, size_t max_output, int *exit_code) {
    if (exit_code) *exit_code = -1;
    FILE *fp = popen(cmd, "r");
    if (!fp) return strdup("Error: failed to execute command");

    StringBuf sb;
    sb_init(&sb);
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) {
        size_t take = n;
        if (sb.len + take > max_output) take = max_output - sb.len;
        if (take > 0) sb_append(&sb, buf, take);
        if (sb.len >= max_output) {
            /* Drain the remainder so the child isn't blocked writing into a
             * full pipe (which would make pclose hang). */
            while (fread(buf, 1, sizeof(buf), fp) > 0) { /* discard */ }
            break;
        }
    }
    int status = pclose(fp);
    if (exit_code) {
        if (WIFEXITED(status))        *exit_code = WEXITSTATUS(status);
        else if (WIFSIGNALED(status)) *exit_code = -WTERMSIG(status);
        else                          *exit_code = -1;
    }

    if (sb.len == 0) {
        sb_free(&sb);
        return strdup("");
    }
    return sb_to_str(&sb);
}

char *run_command(const char *cmd, size_t max_output) {
    return run_command_status(cmd, max_output, NULL);
}

/* Read ppid (field 4) from /proc/<pid>/stat — robust to a comm with spaces or
 * parens by scanning past the last ')'. Returns -1 on failure. */
static pid_t read_ppid(pid_t pid) {
    char path[64];
    snprintf(path, sizeof path, "/proc/%d/stat", (int)pid);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[1024];
    char *got = fgets(line, sizeof line, f);
    fclose(f);
    if (!got) return -1;
    char *rp = strrchr(line, ')');
    if (!rp) return -1;
    char st; int ppid = -1;
    if (sscanf(rp + 1, " %c %d", &st, &ppid) < 2) return -1;
    return (pid_t)ppid;
}

/* Collect all descendant PIDs of `root` (excluding root) by scanning /proc and
 * following PPID links. Call BEFORE killing anything, while the tree is intact —
 * this catches children a shell put in their OWN process group via `cmd &`, which
 * a single killpg() would miss. Returns the count written to out[] (up to cap). */
static int collect_descendants(pid_t root, pid_t *out, int cap) {
    static pid_t pids[8192], ppids[8192];
    static char  isdesc[8192];
    int n = 0;
    DIR *d = opendir("/proc");
    if (!d) return 0;
    struct dirent *e;
    while ((e = readdir(d)) && n < 8192) {
        int allnum = (e->d_name[0] != '\0');
        for (const char *c = e->d_name; *c; c++) if (*c < '0' || *c > '9') { allnum = 0; break; }
        if (!allnum) continue;
        pid_t p = (pid_t)atoi(e->d_name);
        pid_t pp = read_ppid(p);
        if (pp < 0) continue;
        pids[n] = p; ppids[n] = pp; isdesc[n] = 0; n++;
    }
    closedir(d);
    /* iterate to a fixed point: a node is a descendant if its parent is root or
     * an already-marked descendant */
    int changed = 1;
    while (changed) {
        changed = 0;
        for (int i = 0; i < n; i++) {
            if (isdesc[i]) continue;
            int mark = (ppids[i] == root);
            if (!mark)
                for (int j = 0; j < n; j++)
                    if (isdesc[j] && pids[j] == ppids[i]) { mark = 1; break; }
            if (mark) { isdesc[i] = 1; changed = 1; }
        }
    }
    int count = 0;
    for (int i = 0; i < n && count < cap; i++) if (isdesc[i]) out[count++] = pids[i];
    return count;
}

/* Run a shell command with a wall-clock timeout. The child runs in its own
 * process group; on expiry the WHOLE group is killed (SIGTERM, then SIGKILL
 * after a short grace) so a runaway descendant (e.g. a hung test runner or an
 * exponential solution) can't outlive the call. Output is captured up to
 * max_output bytes (the rest is drained so the child never blocks on a full
 * pipe). Sets *timed_out=1 iff the deadline was hit. Caller frees. */
char *run_command_timeout(const char *cmd, size_t max_output, int timeout_s,
                          int *timed_out) {
    if (timed_out) *timed_out = 0;
    if (timeout_s <= 0) timeout_s = 120;

    int pfd[2];
    if (pipe(pfd) != 0) return strdup("Error: pipe failed");

    pid_t pid = fork();
    if (pid < 0) {
        close(pfd[0]); close(pfd[1]);
        return strdup("Error: fork failed");
    }
    if (pid == 0) {
        /* child: own process group, stdout+stderr -> pipe, exec the shell */
        setpgid(0, 0);
        close(pfd[0]);
        dup2(pfd[1], STDOUT_FILENO);
        dup2(pfd[1], STDERR_FILENO);
        close(pfd[1]);
        execl("/bin/sh", "sh", "-c", cmd, (char *)NULL);
        _exit(127);
    }

    /* parent */
    close(pfd[1]);
    setpgid(pid, pid);                       /* race-safe; ignore errors */

    StringBuf sb; sb_init(&sb);
    char buf[4096];
    time_t deadline = time(NULL) + timeout_s;
    int hit_timeout = 0;

    for (;;) {
        long remaining = (long)deadline - (long)time(NULL);
        if (remaining <= 0) { hit_timeout = 1; break; }
        struct pollfd p = { pfd[0], POLLIN, 0 };
        int pr = poll(&p, 1, remaining > 1 ? 1000 : (int)(remaining * 1000));
        if (pr < 0) { if (errno == EINTR) continue; break; }
        if (pr == 0) continue;               /* idle; re-check the deadline */
        ssize_t n = read(pfd[0], buf, sizeof buf);
        if (n <= 0) break;                   /* EOF: child finished */
        if (sb.len < max_output) {
            size_t take = (size_t)n;
            if (sb.len + take > max_output) take = max_output - sb.len;
            sb_append(&sb, buf, take);
        }
        /* output beyond max_output is read and discarded above the cap */
    }
    close(pfd[0]);

    if (hit_timeout) {
        if (timed_out) *timed_out = 1;
        pid_t pgid = pid;                     /* child is its own group leader */
        /* snapshot the descendant tree BEFORE killing (so `cmd &` children that
         * escaped into their own process group are still reachable by PID) */
        pid_t desc[1024];
        int nd = collect_descendants(pid, desc, 1024);
        kill(-pgid, SIGTERM);
        for (int i = 0; i < nd; i++) kill(desc[i], SIGTERM);
        int reaped = 0;
        for (int i = 0; i < 30; i++) {        /* up to ~3s grace for SIGTERM */
            if (waitpid(pid, NULL, WNOHANG) == pid) { reaped = 1; break; }
            struct timespec ts = { 0, 100L * 1000 * 1000 };
            nanosleep(&ts, NULL);
        }
        kill(-pgid, SIGKILL);                 /* sweep in-group stragglers */
        for (int i = 0; i < nd; i++) kill(desc[i], SIGKILL);  /* and out-of-group ones */
        if (!reaped) waitpid(pid, NULL, 0);
    } else {
        waitpid(pid, NULL, 0);
    }

    if (sb.len == 0) { sb_free(&sb); return strdup(""); }
    return sb_to_str(&sb);
}

/* ── Read a whole file into a malloc'd, NUL-terminated buffer ───────── */
/* Returns NULL on any error (open/seek/tell/alloc). Guards ftell() < 0 so a
 * non-seekable / errored stream can't underflow the malloc size. Caller frees.
 * *out_len (optional) gets the byte count, excluding the appended '\0'. */

char *read_file_all(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, 1, (size_t)sz, f);
    buf[n] = '\0';
    fclose(f);
    if (out_len) *out_len = n;
    return buf;
}

/* ── Recursive mkdir ───────────────────────────────────────────────── */

int mkdir_p(const char *path) {
    char tmp[1024];
    int k = snprintf(tmp, sizeof(tmp), "%s", path);
    if (k < 0 || (size_t)k >= sizeof(tmp)) return -1;  /* refuse a truncated path */
    size_t len = strlen(tmp);
    if (len && tmp[len - 1] == '/') tmp[len - 1] = '\0';
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
    return 0;
}

/* ── Count newlines in file (rewinds the FILE *) ───────────────────── */

size_t count_lines(FILE *f) {
    fseek(f, 0, SEEK_SET);
    size_t count = 0;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        for (size_t i = 0; i < n; i++)
            if (buf[i] == '\n') count++;
    }
    fseek(f, 0, SEEK_SET);
    return count;
}

/* ── Time ──────────────────────────────────────────────────────────── */

double time_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* ── URL encode / decode ───────────────────────────────────────────── */

char *url_encode(const char *input) {
    static const char hex[] = "0123456789ABCDEF";
    StringBuf sb;
    sb_init(&sb);
    for (const char *p = input; *p; p++) {
        unsigned char c = *p;
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~') {
            sb_append_char(&sb, c);
        } else if (c == ' ') {
            sb_append_char(&sb, '+');
        } else {
            sb_append_char(&sb, '%');
            sb_append_char(&sb, hex[c >> 4]);
            sb_append_char(&sb, hex[c & 0x0F]);
        }
    }
    return sb_to_str(&sb);
}

char *url_decode(const char *input) {
    StringBuf sb;
    sb_init(&sb);
    for (const char *p = input; *p; p++) {
        if (*p == '%' && p[1] && p[2]) {
            char h[3] = { p[1], p[2], 0 };
            unsigned int byte;
            if (sscanf(h, "%x", &byte) == 1) {
                sb_append_char(&sb, (char)byte);
                p += 2;
            } else {
                sb_append_char(&sb, *p);
            }
        } else if (*p == '+') {
            sb_append_char(&sb, ' ');
        } else {
            sb_append_char(&sb, *p);
        }
    }
    return sb_to_str(&sb);
}

/* ── JSON escape ───────────────────────────────────────────────────── */

void json_escape_into(StringBuf *sb, const char *s) {
    sb_append_char(sb, '"');
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char c = *p;
        switch (c) {
            case '"':  sb_append_str(sb, "\\\""); break;
            case '\\': sb_append_str(sb, "\\\\"); break;
            case '\n': sb_append_str(sb, "\\n");  break;
            case '\r': sb_append_str(sb, "\\r");  break;
            case '\t': sb_append_str(sb, "\\t");  break;
            case '\b': sb_append_str(sb, "\\b");  break;
            case '\f': sb_append_str(sb, "\\f");  break;
            default:
                if (c < 0x20) {
                    char esc[8];
                    snprintf(esc, sizeof(esc), "\\u%04x", c);
                    sb_append_str(sb, esc);
                } else {
                    sb_append_char(sb, (char)c);
                }
        }
    }
    sb_append_char(sb, '"');
}

/* ── Minimal JSON path extractor ──────────────────────────────────── */
/* Supports paths like "result.contents.value" or "results.0.url".
 * Assumes input is well-formed JSON. Lifted from lsp.c so web.c (SearXNG
 * JSON) and the LSP client share one parser. */

static const char *jx_skip_ws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

static const char *jx_skip_value(const char *p);

static const char *jx_skip_string(const char *p) {
    if (*p != '"') return NULL;
    p++;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) p += 2;
        else p++;
    }
    return *p == '"' ? p + 1 : NULL;
}

static const char *jx_skip_object(const char *p) {
    if (*p != '{') return NULL;
    p++;
    p = jx_skip_ws(p);
    if (*p == '}') return p + 1;
    while (*p) {
        p = jx_skip_ws(p);
        p = jx_skip_string(p);
        if (!p) return NULL;
        p = jx_skip_ws(p);
        if (*p != ':') return NULL;
        p++;
        p = jx_skip_ws(p);
        p = jx_skip_value(p);
        if (!p) return NULL;
        p = jx_skip_ws(p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') return p + 1;
        return NULL;
    }
    return NULL;
}

static const char *jx_skip_array(const char *p) {
    if (*p != '[') return NULL;
    p++;
    p = jx_skip_ws(p);
    if (*p == ']') return p + 1;
    while (*p) {
        p = jx_skip_ws(p);
        p = jx_skip_value(p);
        if (!p) return NULL;
        p = jx_skip_ws(p);
        if (*p == ',') { p++; continue; }
        if (*p == ']') return p + 1;
        return NULL;
    }
    return NULL;
}

static const char *jx_skip_value(const char *p) {
    p = jx_skip_ws(p);
    if (*p == '"') return jx_skip_string(p);
    if (*p == '{') return jx_skip_object(p);
    if (*p == '[') return jx_skip_array(p);
    while (*p && *p != ',' && *p != '}' && *p != ']' &&
           *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
    return p;
}

/* In an object, find the value start of a given key. Returns NULL if missing. */
static const char *jx_find_key(const char *p, const char *key, size_t klen) {
    p = jx_skip_ws(p);
    if (*p != '{') return NULL;
    p++;
    while (*p) {
        p = jx_skip_ws(p);
        if (*p == '}') return NULL;
        if (*p != '"') return NULL;
        p++;
        const char *k_start = p;
        while (*p && *p != '"') {
            if (*p == '\\' && p[1]) p += 2;
            else p++;
        }
        size_t k_len = (size_t)(p - k_start);
        if (*p == '"') p++;
        p = jx_skip_ws(p);
        if (*p != ':') return NULL;
        p++;
        p = jx_skip_ws(p);
        if (k_len == klen && memcmp(k_start, key, klen) == 0) return p;
        p = jx_skip_value(p);
        if (!p) return NULL;
        p = jx_skip_ws(p);
        if (*p == ',') { p++; continue; }
        return NULL;
    }
    return NULL;
}

/* In an array, return value start of element at index, or NULL. */
static const char *jx_array_index(const char *p, int idx) {
    p = jx_skip_ws(p);
    if (*p != '[') return NULL;
    p++;
    int i = 0;
    while (*p) {
        p = jx_skip_ws(p);
        if (*p == ']') return NULL;
        if (i == idx) return p;
        p = jx_skip_value(p);
        if (!p) return NULL;
        p = jx_skip_ws(p);
        if (*p == ',') { p++; i++; continue; }
        if (*p == ']') return NULL;
        return NULL;
    }
    return NULL;
}

/* Walk a dotted path. Numeric segments are treated as array indices. */
static const char *jx_walk(const char *json, const char *path) {
    const char *p = json;
    while (*path) {
        const char *dot = strchr(path, '.');
        size_t plen = dot ? (size_t)(dot - path) : strlen(path);

        bool is_num = true;
        for (size_t i = 0; i < plen; i++)
            if (path[i] < '0' || path[i] > '9') { is_num = false; break; }

        if (is_num) {
            int idx = 0;
            for (size_t i = 0; i < plen; i++) idx = idx * 10 + (path[i] - '0');
            p = jx_array_index(p, idx);
        } else {
            p = jx_find_key(p, path, plen);
        }
        if (!p) return NULL;
        path += plen;
        if (*path == '.') path++;
    }
    return p;
}

/* If p points at a JSON string value, decode it to a heap C string. */
static char *jx_decode_string(const char *p) {
    p = jx_skip_ws(p);
    if (*p != '"') return NULL;
    p++;
    StringBuf out;
    sb_init(&out);
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) {
            switch (p[1]) {
                case 'n':  sb_append_char(&out, '\n'); p += 2; break;
                case 't':  sb_append_char(&out, '\t'); p += 2; break;
                case 'r':  sb_append_char(&out, '\r'); p += 2; break;
                case '\\': sb_append_char(&out, '\\'); p += 2; break;
                case '"':  sb_append_char(&out, '"');  p += 2; break;
                case '/':  sb_append_char(&out, '/');  p += 2; break;
                case 'u':  sb_append_char(&out, '?');  p += 6; break;  /* skip unicode v1 */
                default:   sb_append_char(&out, p[1]); p += 2; break;
            }
        } else {
            sb_append_char(&out, *p);
            p++;
        }
    }
    return sb_to_str(&out);
}

char *jx_get_string(const char *json, const char *path) {
    const char *p = jx_walk(json, path);
    return p ? jx_decode_string(p) : NULL;
}

long jx_get_int(const char *json, const char *path) {
    const char *p = jx_walk(json, path);
    if (!p) return -1;
    p = jx_skip_ws(p);
    int sign = 1;
    if (*p == '-') { sign = -1; p++; }
    long n = 0;
    while (*p >= '0' && *p <= '9') { n = n * 10 + (*p - '0'); p++; }
    return n * sign;
}

bool jx_has_key(const char *json, const char *key) {
    return jx_walk(json, key) != NULL;
}
