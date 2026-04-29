#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>

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
    sb->data = realloc(sb->data, newcap);
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
                al.args = realloc(al.args, cap * sizeof(char *));
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

char *run_command(const char *cmd, size_t max_output) {
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
        if (sb.len >= max_output) break;
    }
    pclose(fp);

    if (sb.len == 0) {
        sb_free(&sb);
        return strdup("");
    }
    return sb_to_str(&sb);
}

/* ── Recursive mkdir ───────────────────────────────────────────────── */

int mkdir_p(const char *path) {
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", path);
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
