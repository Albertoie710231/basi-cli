#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <poll.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>

#include "util.h"
#include "lsp.h"

/* ── LSP client (clangd) backing the code_context tool ─────────────── */
/* Deterministic-first: a structured tool that hands the LLM precise
 * type/signature info via clangd, instead of asking it to extract
 * structure from raw text. Hand-rolled minimal JSON path-extractor
 * keeps the codebase single-file (no vendored JSON library). */

#define LSP_MAX_OPENED 64

typedef struct {
    pid_t pid;
    int   in_fd;            /* write to clangd stdin  */
    int   out_fd;           /* read  from clangd stdout */
    int   next_id;
    bool  initialized;
    char *opened[LSP_MAX_OPENED];   /* URIs we've sent didOpen for */
    int   n_opened;
} LspClient;

static LspClient lsp = { -1, -1, -1, 1, false, { NULL }, 0 };

/* ── DEBUG: opt-in via BASI_LSP_DEBUG=1 — writes RPC traffic to /tmp/basi-lsp.log ── */
static FILE *lsp_dbg_fp = NULL;
static bool  lsp_dbg_inited = false;
static void lsp_dbg_init(void) {
    if (lsp_dbg_inited) return;
    lsp_dbg_inited = true;
    if (getenv("BASI_LSP_DEBUG")) {
        lsp_dbg_fp = fopen("/tmp/basi-lsp.log", "w");
        if (lsp_dbg_fp) {
            fprintf(lsp_dbg_fp, "=== BASI LSP debug log (pid %d) ===\n", (int)getpid());
            fflush(lsp_dbg_fp);
        }
    }
}
static void lsp_dbg(const char *fmt, ...) {
    if (!lsp_dbg_fp) return;
    va_list ap; va_start(ap, fmt);
    vfprintf(lsp_dbg_fp, fmt, ap);
    va_end(ap);
    fflush(lsp_dbg_fp);
}

/* ── Minimal JSON path extractor ──────────────────────────────────── */
/* Supports paths like "result.contents.value" or "result.0.uri".
 * Assumes input is well-formed JSON (LSP responses always are). */

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

static char *jx_get_string(const char *json, const char *path) {
    const char *p = jx_walk(json, path);
    return p ? jx_decode_string(p) : NULL;
}

static long jx_get_int(const char *json, const char *path) {
    const char *p = jx_walk(json, path);
    if (!p) return -1;
    p = jx_skip_ws(p);
    int sign = 1;
    if (*p == '-') { sign = -1; p++; }
    long n = 0;
    while (*p >= '0' && *p <= '9') { n = n * 10 + (*p - '0'); p++; }
    return n * sign;
}

static bool jx_has_key(const char *json, const char *key) {
    return jx_walk(json, key) != NULL;
}

/* ── JSON-RPC framing over the clangd pipes ───────────────────────── */

static bool rpc_write(int fd, const char *body, size_t blen) {
    char hdr[64];
    int hlen = snprintf(hdr, sizeof(hdr), "Content-Length: %zu\r\n\r\n", blen);
    if (hlen <= 0) return false;
    lsp_dbg("[OUT fd=%d] header(%d): Content-Length: %zu\n[OUT body %zu bytes]: %.500s%s\n",
            fd, hlen, blen, blen, body, blen > 500 ? "...(truncated)" : "");
    size_t off = 0;
    while (off < (size_t)hlen) {
        ssize_t w = write(fd, hdr + off, hlen - off);
        if (w < 0) { if (errno == EINTR) continue; lsp_dbg("[OUT] write hdr failed: %s\n", strerror(errno)); return false; }
        off += w;
    }
    off = 0;
    while (off < blen) {
        ssize_t w = write(fd, body + off, blen - off);
        if (w < 0) { if (errno == EINTR) continue; lsp_dbg("[OUT] write body failed: %s\n", strerror(errno)); return false; }
        off += w;
    }
    return true;
}

/* Wait up to timeout_ms for fd to become readable. Returns 1=ready, 0=timeout, -1=error. */
static int wait_readable(int fd, int timeout_ms) {
    struct pollfd pfd = { .fd = fd, .events = POLLIN, .revents = 0 };
    int rv;
    do { rv = poll(&pfd, 1, timeout_ms); } while (rv < 0 && errno == EINTR);
    return rv;
}

static char *rpc_read(int fd, int timeout_ms) {
    /* Read header until \r\n\r\n */
    char hdr[1024];
    size_t n = 0;
    while (n + 1 < sizeof(hdr)) {
        int rv = wait_readable(fd, timeout_ms);
        if (rv == 0) { lsp_dbg("[IN  fd=%d] timeout reading header at byte %zu (timeout=%dms)\n", fd, n, timeout_ms); return NULL; }
        if (rv < 0)  { lsp_dbg("[IN  fd=%d] poll error: %s\n", fd, strerror(errno)); return NULL; }
        char c;
        ssize_t r = read(fd, &c, 1);
        if (r < 0) { if (errno == EINTR) continue; lsp_dbg("[IN  fd=%d] read error: %s\n", fd, strerror(errno)); return NULL; }
        if (r == 0) { lsp_dbg("[IN  fd=%d] EOF reading header at byte %zu (header so far: %.*s)\n", fd, n, (int)n, hdr); return NULL; }
        hdr[n++] = c;
        if (n >= 4 && hdr[n-4] == '\r' && hdr[n-3] == '\n' &&
            hdr[n-2] == '\r' && hdr[n-1] == '\n') break;
    }
    hdr[n] = '\0';
    lsp_dbg("[IN  fd=%d] header(%zu): %.*s", fd, n, (int)(n - 4), hdr);

    const char *cl = strstr(hdr, "Content-Length:");
    if (!cl) { lsp_dbg("[IN ] no Content-Length in header\n"); return NULL; }
    cl += 15;
    while (*cl == ' ') cl++;
    long blen = atol(cl);
    if (blen <= 0 || blen > 32 * 1024 * 1024) { lsp_dbg("[IN ] invalid Content-Length: %ld\n", blen); return NULL; }

    char *body = malloc(blen + 1);
    if (!body) return NULL;
    size_t off = 0;
    while (off < (size_t)blen) {
        int rv = wait_readable(fd, timeout_ms);
        if (rv <= 0) { lsp_dbg("[IN ] timeout/error reading body at offset %zu/%ld\n", off, blen); free(body); return NULL; }
        ssize_t r = read(fd, body + off, blen - off);
        if (r < 0) { if (errno == EINTR) continue; free(body); return NULL; }
        if (r == 0) { lsp_dbg("[IN ] EOF reading body at offset %zu/%ld\n", off, blen); free(body); return NULL; }
        off += r;
    }
    body[blen] = '\0';
    lsp_dbg("[IN  fd=%d] body(%ld): %.500s%s\n", fd, blen, body, blen > 500 ? "...(truncated)" : "");
    return body;
}

/* ── Subprocess: spawn clangd with bidirectional pipes ────────────── */

static bool lsp_spawn(LspClient *c) {
    int in_pipe[2], out_pipe[2];
    if (pipe(in_pipe) < 0) return false;
    if (pipe(out_pipe) < 0) {
        close(in_pipe[0]); close(in_pipe[1]);
        return false;
    }
    pid_t pid = fork();
    if (pid < 0) {
        close(in_pipe[0]); close(in_pipe[1]);
        close(out_pipe[0]); close(out_pipe[1]);
        return false;
    }
    if (pid == 0) {
        /* child: clangd. Redirect stderr to /dev/null normally; to a debug
         * log file when BASI_LSP_DEBUG=1 so we can see clangd's own
         * diagnostics when something goes wrong. */
        dup2(in_pipe[0],  STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);
        const char *stderr_path = getenv("BASI_LSP_DEBUG")
            ? "/tmp/basi-clangd-stderr.log" : "/dev/null";
        int errfd = open(stderr_path,
            getenv("BASI_LSP_DEBUG") ? (O_WRONLY | O_CREAT | O_TRUNC) : O_WRONLY,
            0644);
        if (errfd >= 0) {
            dup2(errfd, STDERR_FILENO);
            close(errfd);
        }
        close(in_pipe[0]); close(in_pipe[1]);
        close(out_pipe[0]); close(out_pipe[1]);
        /* Also use --log=verbose under debug so clangd reports more */
        const char *log_arg = getenv("BASI_LSP_DEBUG") ? "--log=verbose" : "--log=error";
        execlp("clangd", "clangd", log_arg, "--background-index=false", NULL);
        _exit(127);
    }
    /* parent */
    close(in_pipe[0]);
    close(out_pipe[1]);
    c->pid = pid;
    c->in_fd = in_pipe[1];
    c->out_fd = out_pipe[0];
    c->next_id = 1;
    c->initialized = false;
    c->n_opened = 0;
    return true;
}

static void lsp_kill(LspClient *c) {
    if (c->pid <= 0) return;
    if (c->in_fd >= 0) close(c->in_fd);
    if (c->out_fd >= 0) close(c->out_fd);
    kill(c->pid, SIGTERM);
    int status;
    waitpid(c->pid, &status, 0);
    c->pid = -1; c->in_fd = -1; c->out_fd = -1;
    c->initialized = false;
    for (int i = 0; i < c->n_opened; i++) free(c->opened[i]);
    c->n_opened = 0;
}

/* Send request and wait for matching response (skipping notifications,
 * auto-replying empty results to server-to-client requests so clangd
 * doesn't block waiting on us). 60s ceiling per response.
 *
 * params + params_len = explicit length, NEVER strlen. Past bug: relying on
 * strlen of a StringBuf buffer let trailing garbage from realloc'd memory
 * leak past the null terminator and into the wire, breaking clangd. */
#define LSP_TIMEOUT_MS 60000

void lsp_shutdown(void) {
    lsp_kill(&lsp);
}

static char *lsp_request(LspClient *c, const char *method,
                         const char *params, size_t params_len) {
    int id = c->next_id++;
    StringBuf body;
    sb_init(&body);
    char prefix[256];
    int plen = snprintf(prefix, sizeof(prefix),
        "{\"jsonrpc\":\"2.0\",\"id\":%d,\"method\":\"%s\",\"params\":", id, method);
    sb_append(&body, prefix, (size_t)plen);
    sb_append(&body, params, params_len);
    sb_append_char(&body, '}');

    if (!rpc_write(c->in_fd, body.data, body.len)) {
        sb_free(&body);
        return NULL;
    }
    sb_free(&body);

    char idneedle[32];
    snprintf(idneedle, sizeof(idneedle), "\"id\":%d", id);
    while (1) {
        char *resp = rpc_read(c->out_fd, LSP_TIMEOUT_MS);
        if (!resp) return NULL;  /* timeout or pipe death — caller treats as failure */

        if (strstr(resp, idneedle)) return resp;

        /* Server-to-client request? (has "method" AND "id"). Reply empty so
         * clangd unblocks. Otherwise it's a notification — just drop. */
        if (jx_has_key(resp, "method") && jx_has_key(resp, "id")) {
            long srv_id = jx_get_int(resp, "id");
            if (srv_id >= 0) {
                char reply[128];
                int rlen = snprintf(reply, sizeof(reply),
                    "{\"jsonrpc\":\"2.0\",\"id\":%ld,\"result\":null}", srv_id);
                rpc_write(c->in_fd, reply, (size_t)rlen);
            }
        }
        free(resp);
    }
}

static bool lsp_notify(LspClient *c, const char *method,
                       const char *params, size_t params_len) {
    StringBuf body;
    sb_init(&body);
    char prefix[256];
    int plen = snprintf(prefix, sizeof(prefix),
        "{\"jsonrpc\":\"2.0\",\"method\":\"%s\",\"params\":", method);
    sb_append(&body, prefix, (size_t)plen);
    sb_append(&body, params, params_len);
    sb_append_char(&body, '}');
    /* Diagnostic: dump first 80 + last 30 bytes of body so we can verify
     * what actually gets written, not just trust strlen / null terminator. */
    if (lsp_dbg_fp) {
        lsp_dbg("[notify] params_len=%zu body.len=%zu\n", params_len, body.len);
        lsp_dbg("[notify] body[0..80]: ");
        size_t first = body.len < 80 ? body.len : 80;
        for (size_t i = 0; i < first; i++) lsp_dbg("%c", body.data[i] >= 0x20 && body.data[i] < 0x7f ? body.data[i] : '.');
        lsp_dbg("\n[notify] body[last 30] hex: ");
        size_t start = body.len > 30 ? body.len - 30 : 0;
        for (size_t i = start; i < body.len; i++) lsp_dbg("%02x ", (unsigned char)body.data[i]);
        lsp_dbg("\n");
    }
    bool ok = rpc_write(c->in_fd, body.data, body.len);
    sb_free(&body);
    return ok;
}

/* ── LSP handshake + textDocument lifecycle ───────────────────────── */

static bool lsp_initialize(LspClient *c) {
    char cwd[1024];
    if (!getcwd(cwd, sizeof(cwd))) return false;

    char buf[2048];
    int n = snprintf(buf, sizeof(buf),
        "{\"processId\":%d,\"rootUri\":\"file://%s\",\"capabilities\":"
        "{\"textDocument\":{\"hover\":{\"contentFormat\":[\"plaintext\"]},\"definition\":{}}}}",
        (int)getpid(), cwd);

    char *resp = lsp_request(c, "initialize", buf, (size_t)n);
    if (!resp) return false;
    bool ok = !strstr(resp, "\"error\"");
    free(resp);
    if (!ok) return false;

    static const char empty_obj[] = "{}";
    if (!lsp_notify(c, "initialized", empty_obj, sizeof(empty_obj) - 1)) return false;
    c->initialized = true;
    return true;
}

/* JSON-escape a string into a StringBuf (NO surrounding quotes — caller adds them). */
static void lsp_json_escape(StringBuf *out, const char *s) {
    for (const char *p = s; *p; p++) {
        unsigned char b = (unsigned char)*p;
        if      (*p == '"')  sb_append_str(out, "\\\"");
        else if (*p == '\\') sb_append_str(out, "\\\\");
        else if (*p == '\n') sb_append_str(out, "\\n");
        else if (*p == '\r') sb_append_str(out, "\\r");
        else if (*p == '\t') sb_append_str(out, "\\t");
        else if (b < 0x20)   { char hx[8]; snprintf(hx, sizeof(hx), "\\u%04x", b); sb_append_str(out, hx); }
        else                 sb_append_char(out, *p);
    }
}

static char *abs_uri(const char *path) {
    char abspath[2048];
    if (path[0] == '/') {
        snprintf(abspath, sizeof(abspath), "%.2000s", path);
    } else {
        char cwd[1024];
        if (!getcwd(cwd, sizeof(cwd))) return NULL;
        snprintf(abspath, sizeof(abspath), "%.1000s/%.1000s", cwd, path);
    }
    size_t ulen = strlen(abspath) + 8;
    char *uri = malloc(ulen);
    snprintf(uri, ulen, "file://%s", abspath);
    return uri;
}

/* didOpen on first sight, didChange on subsequent calls. */
static bool lsp_sync_file(LspClient *c, const char *path, const char *content) {
    char *uri = abs_uri(path);
    if (!uri) return false;

    bool first = true;
    for (int i = 0; i < c->n_opened; i++)
        if (strcmp(c->opened[i], uri) == 0) { first = false; break; }

    StringBuf p;
    sb_init(&p);
    if (first) {
        sb_append_str(&p, "{\"textDocument\":{\"uri\":\"");
        lsp_json_escape(&p, uri);
        sb_append_str(&p, "\",\"languageId\":\"c\",\"version\":1,\"text\":\"");
        lsp_json_escape(&p, content);
        sb_append_str(&p, "\"}}");
        bool ok = lsp_notify(c, "textDocument/didOpen", p.data, p.len);
        sb_free(&p);
        if (ok && c->n_opened < LSP_MAX_OPENED)
            c->opened[c->n_opened++] = uri;
        else
            free(uri);
        return ok;
    } else {
        sb_append_str(&p, "{\"textDocument\":{\"uri\":\"");
        lsp_json_escape(&p, uri);
        sb_append_str(&p, "\",\"version\":2},\"contentChanges\":[{\"text\":\"");
        lsp_json_escape(&p, content);
        sb_append_str(&p, "\"}]}");
        bool ok = lsp_notify(c, "textDocument/didChange", p.data, p.len);
        sb_free(&p);
        free(uri);
        return ok;
    }
}

/* C keywords to skip when picking a hover column. clangd returns null for
 * hover on type/storage keywords, so first-non-whitespace lands wrong on
 * lines like `static char *foo(...)`. We walk past these to find the first
 * real identifier. */
static const char *const C_HOVER_SKIP_KEYWORDS[] = {
    "static", "inline", "extern", "const", "volatile", "register", "restrict",
    "struct", "union", "enum", "typedef", "auto",
    "int", "char", "void", "bool", "float", "double", "long", "short",
    "unsigned", "signed", "size_t", "ssize_t", "uint8_t", "uint16_t",
    "uint32_t", "uint64_t", "int8_t", "int16_t", "int32_t", "int64_t",
    "if", "else", "for", "while", "do", "switch", "case", "default",
    "return", "break", "continue", "goto", "sizeof",
    NULL
};

static bool is_c_skip_keyword(const char *s, size_t len) {
    for (const char *const *kw = C_HOVER_SKIP_KEYWORDS; *kw; kw++) {
        if (strlen(*kw) == len && memcmp(*kw, s, len) == 0) return true;
    }
    return false;
}

/* Find the column of the first non-keyword identifier on the line.
 * Falls back to first non-whitespace column if none is found. */
static int find_hover_column(const char *line) {
    const char *p = line;
    int fallback = 0;
    while (*p == ' ' || *p == '\t') { p++; fallback++; }
    while (*p && *p != '\n') {
        if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') || *p == '_') {
            const char *id = p;
            while ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
                   (*p >= '0' && *p <= '9') || *p == '_') p++;
            size_t len = (size_t)(p - id);
            if (!is_c_skip_keyword(id, len)) return (int)(id - line);
            while (*p == ' ' || *p == '\t' || *p == '*') p++;
            continue;
        }
        p++;
    }
    return fallback;
}

/* lsp_hover return contract:
 *   NULL          → transport failure (timeout/pipe death). Caller should kill clangd.
 *   ""  (empty)   → clangd responded with result:null (no symbol at position). Don't kill.
 *   "..."         → real hover text (signature, type, docstring). */
static char *lsp_hover(LspClient *c, const char *path, int line, int col) {
    char *uri = abs_uri(path);
    if (!uri) return NULL;
    char buf[4096];
    int n = snprintf(buf, sizeof(buf),
        "{\"textDocument\":{\"uri\":\"%s\"},\"position\":{\"line\":%d,\"character\":%d}}",
        uri, line, col);
    free(uri);
    char *resp = lsp_request(c, "textDocument/hover", buf, (size_t)n);
    if (!resp) return NULL;  /* transport failure */
    /* hover result shape: result.contents.value (markup) OR result.contents (string in older spec) */
    char *value = jx_get_string(resp, "result.contents.value");
    if (!value) value = jx_get_string(resp, "result.contents");
    free(resp);
    return value ? value : strdup("");  /* empty = response ok, no info at position */
}

/* ── code_context orchestrator ────────────────────────────────────── */
/* Semantic-only tool: returns ONLY clangd's structural info for a symbol,
 * fenced so the model can't conflate it with surrounding text. Accepts
 * either <file>:<line> or <file> <symbol> (the latter is robust to file
 * edits — internally grep for the symbol's definition line). */

static bool is_id_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
}

/* Find the line number (1-based) where <symbol> appears as a top-level
 * definition. Heuristic: line starts at column 0, <symbol> appears with
 * word boundaries, followed by '(' / '=' / ';' / ',' / EOL after optional
 * whitespace. Returns -1 if not found. */
static int find_symbol_definition_line(const char *file_path, const char *symbol) {
    FILE *f = fopen(file_path, "r");
    if (!f) return -1;
    char line[4096];
    int line_num = 0;
    size_t slen = strlen(symbol);

    while (fgets(line, sizeof(line), f)) {
        line_num++;
        if (line[0] == ' ' || line[0] == '\t') continue;  /* indented = function body */

        for (char *p = line; (p = strstr(p, symbol)) != NULL; p++) {
            bool prev_ok = (p == line) || !is_id_char(*(p - 1));
            bool next_ok = !is_id_char(p[slen]);
            if (!prev_ok || !next_ok) continue;
            const char *q = p + slen;
            while (*q == ' ' || *q == '\t') q++;
            if (*q == '(' || *q == '=' || *q == ';' || *q == ',' ||
                *q == '\n' || *q == '\0') {
                fclose(f);
                return line_num;
            }
        }
    }
    fclose(f);
    return -1;
}

char *execute_code_context(const char *args) {
    while (*args == ' ' || *args == '\t' || *args == '\n') args++;
    if (!*args) {
        return strdup("Error: code_context requires <file>:<line> or <file> <symbol>. "
                      "Examples: <tool>code_context src/main.c execute_apply_patch</tool> "
                      "(by symbol — preferred), or <tool>code_context src/main.c:2102</tool>");
    }

    /* Parse: STRICTLY <file> <symbol> separated by whitespace. One form, no
     * fallbacks, no colon syntax. Loud, specific errors when the model deviates. */
    char *file = NULL;
    char *symbol = NULL;
    {
        const char *sep = strpbrk(args, " \t");
        if (!sep) {
            return strdup("Error: code_context expects exactly two whitespace-separated arguments: <file> <symbol>. "
                          "Example: <tool>code_context src/main.c execute_apply_patch</tool>");
        }
        size_t flen = (size_t)(sep - args);
        if (flen == 0) {
            return strdup("Error: code_context: empty file path before whitespace.");
        }
        file = malloc(flen + 1);
        memcpy(file, args, flen);
        file[flen] = '\0';

        const char *s = sep;
        while (*s == ' ' || *s == '\t') s++;
        const char *e = s;
        while (*e && *e != ' ' && *e != '\t' && *e != '\n') e++;
        size_t slen = (size_t)(e - s);
        if (slen == 0) {
            free(file);
            return strdup("Error: code_context: missing symbol name after the file path.");
        }
        /* Symbol must be a C identifier — reject digits-first (line numbers) and any
         * non-identifier characters (parens, colons, etc.). */
        bool bad_start = (s[0] >= '0' && s[0] <= '9');
        bool bad_char  = false;
        if (!bad_start) {
            for (size_t i = 0; i < slen; i++) {
                if (!is_id_char(s[i])) { bad_char = true; break; }
            }
        }
        if (bad_start || bad_char) {
            char *err = malloc(512);
            snprintf(err, 512,
                "Error: code_context: '%.100s' is not a valid C identifier. "
                "Pass only the symbol's name (no line numbers, no colons, no parens). "
                "Example: <tool>code_context src/main.c execute_apply_patch</tool>",
                s);
            free(file);
            return err;
        }
        symbol = malloc(slen + 1);
        memcpy(symbol, s, slen);
        symbol[slen] = '\0';
    }

    /* Find the definition line via grep-style scan. */
    int line_1based = find_symbol_definition_line(file, symbol);
    if (line_1based < 0) {
        char *err = malloc(512);
        snprintf(err, 512,
            "code_context: symbol '%.100s' not found as a top-level definition in '%.200s'. "
            "Check spelling, or grep to confirm it's defined in this file.",
            symbol, file);
        free(file); free(symbol);
        return err;
    }
    int line0 = line_1based - 1;

    /* Read file content. */
    FILE *f = fopen(file, "r");
    if (!f) {
        char *e = malloc(512);
        snprintf(e, 512, "code_context: cannot open '%.300s' (%s)", file, strerror(errno));
        free(file); free(symbol);
        return e;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize <= 0 || fsize > 8 * 1024 * 1024) {
        fclose(f);
        free(file); free(symbol);
        return strdup("code_context: file is empty or larger than 8MB");
    }
    char *content = malloc(fsize + 1);
    size_t nread = fread(content, 1, fsize, f);
    content[nread] = '\0';
    fclose(f);

    int total_lines = 0;
    for (size_t i = 0; i < nread; i++) if (content[i] == '\n') total_lines++;
    if (content[nread - 1] != '\n') total_lines++;
    if (line0 >= total_lines) {
        char *e = malloc(256);
        snprintf(e, 256, "code_context: line %d out of range (file has %d lines)", line_1based, total_lines);
        free(file); free(symbol); free(content);
        return e;
    }

    /* Pick hover column.
     *  - For symbol form: hover at the symbol's exact column on its definition line.
     *  - For file:line form: walk past leading keywords to the first real identifier. */
    int  hover_col = 0;
    bool target_is_blank = false;
    {
        const char *lp = content;
        int cur = 0;
        while (cur < line0 && *lp) { if (*lp == '\n') cur++; lp++; }
        const char *first_ns = lp;
        while (*first_ns == ' ' || *first_ns == '\t') first_ns++;
        target_is_blank = (*first_ns == '\n' || *first_ns == '\0');

        if (!target_is_blank) {
            const char *line_end = strchr(lp, '\n');
            const char *p = strstr(lp, symbol);
            if (p && (!line_end || p < line_end)) {
                hover_col = (int)(p - lp);
            } else {
                hover_col = find_hover_column(lp);
            }
        }
    }

    if (target_is_blank) {
        /* Should be unreachable since find_symbol_definition_line only returns lines
         * with non-whitespace content, but handle defensively. */
        char *err = malloc(512);
        snprintf(err, 512,
            "code_context: definition line %d for symbol '%.100s' in '%.200s' resolved to a blank line. "
            "This usually means find_symbol_definition_line picked the wrong line — file me a bug.",
            line_1based, symbol, file);
        free(file); free(symbol); free(content);
        return err;
    }

    /* Spawn + initialize clangd lazily on first use. */
    if (lsp.pid <= 0) {
        lsp_dbg_init();
        lsp_dbg("=== code_context request: file=%s line=%d symbol=%s ===\n",
                file, line_1based, symbol ? symbol : "(none)");
        printf("\033[90m[Starting clangd... (one-time, can take 10–30s on large codebases)]\033[0m\n");
        fflush(stdout);
        if (!lsp_spawn(&lsp)) {
            free(file); free(symbol); free(content);
            return strdup("code_context: failed to spawn clangd. Is it installed? Try: pacman -S clang  (or: apt install clangd)");
        }
        lsp_dbg("[lsp_spawn] OK pid=%d in_fd=%d out_fd=%d\n", lsp.pid, lsp.in_fd, lsp.out_fd);
        if (!lsp_initialize(&lsp)) {
            lsp_dbg("[lsp_initialize] FAILED\n");
            lsp_kill(&lsp);
            free(file); free(symbol); free(content);
            return strdup("code_context: clangd failed to initialize. Check that clangd is in PATH and the project compiles.");
        }
        lsp_dbg("[lsp_initialize] OK\n");
    }

    if (!lsp_sync_file(&lsp, file, content)) {
        lsp_kill(&lsp);
        free(file); free(symbol); free(content);
        return strdup("code_context: failed to sync file with clangd (clangd likely died — will respawn on next call)");
    }

    char *hover_text = lsp_hover(&lsp, file, line0, hover_col);
    if (hover_text == NULL) lsp_kill(&lsp);

    /* Build SEMANTIC-ONLY response. No surrounding lines (grep -C handles that).
     * Hover text is fenced so the model can't claim non-clangd content as clangd output. */
    StringBuf out;
    sb_init(&out);
    char header[512];
    snprintf(header, sizeof(header), "symbol: %.100s\nfile: %.200s:%d\n", symbol, file, line_1based);
    sb_append_str(&out, header);

    if (hover_text && *hover_text) {
        sb_append_str(&out, "\n=== begin clangd ===\n");
        sb_append_str(&out, hover_text);
        if (hover_text[strlen(hover_text) - 1] != '\n') sb_append_char(&out, '\n');
        sb_append_str(&out, "=== end clangd ===\n");
    } else {
        sb_append_str(&out,
            "\n=== begin clangd ===\n(no symbol info — clangd couldn't resolve the symbol)\n=== end clangd ===\n"
            "Most likely cause: compile_commands.json missing or out of date.\n");
    }
    free(hover_text);

    free(file); free(symbol); free(content);
    return sb_to_str(&out);
}
