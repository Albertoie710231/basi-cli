#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <termios.h>
#include <errno.h>

#include "llama.h"

#include "util.h"
#include "session.h"

/* ── Session persistence ───────────────────────────────────────────── */

/* Returns malloc'd path: ~/.local/share/basi-cli/projects/<encoded-cwd>/
 * with the directory created. NULL on failure. */
char *session_dir_path(void) {
    const char *home = getenv("HOME");
    if (!home) return NULL;
    char cwd[1024];
    if (!getcwd(cwd, sizeof(cwd))) return NULL;

    char enc[1024];
    size_t i = 0;
    for (const char *p = cwd; *p && i + 1 < sizeof(enc); p++)
        enc[i++] = (*p == '/') ? '-' : *p;
    enc[i] = '\0';

    size_t need = strlen(home) + strlen(enc) + 64;
    char *path = malloc(need);
    snprintf(path, need, "%s/.local/share/basi-cli/projects/%s", home, enc);
    if (mkdir_p(path) != 0) { free(path); return NULL; }
    return path;
}

void session_write_record(FILE *fp, const char *role, const char *content) {
    if (!fp) return;
    StringBuf sb;
    sb_init(&sb);
    sb_append_str(&sb, "{\"role\":");
    json_escape_into(&sb, role);
    sb_append_str(&sb, ",\"content\":");
    json_escape_into(&sb, content);
    sb_append_str(&sb, "}\n");
    fwrite(sb.data, 1, sb.len, fp);
    fflush(fp);
    sb_free(&sb);
}

/* Strict parser for our own writer's format: {"role":"X","content":"Y"} */
static int parse_session_line(const char *line, char **out_role, char **out_content) {
    *out_role = NULL;
    *out_content = NULL;
    const char *p = line;
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '{') return -1;
    p++;

    for (int pair = 0; pair < 2; pair++) {
        while (*p == ' ' || *p == ',') p++;
        if (*p != '"') goto fail;
        p++;
        const char *kstart = p;
        while (*p && *p != '"') p++;
        if (*p != '"') goto fail;
        size_t klen = (size_t)(p - kstart);
        char key[16];
        if (klen >= sizeof(key)) goto fail;
        memcpy(key, kstart, klen);
        key[klen] = '\0';
        p++;
        while (*p == ' ' || *p == ':') p++;
        if (*p != '"') goto fail;
        p++;

        StringBuf val;
        sb_init(&val);
        while (*p && *p != '"') {
            if (*p == '\\') {
                p++;
                switch (*p) {
                    case 'n':  sb_append_char(&val, '\n'); break;
                    case 'r':  sb_append_char(&val, '\r'); break;
                    case 't':  sb_append_char(&val, '\t'); break;
                    case 'b':  sb_append_char(&val, '\b'); break;
                    case 'f':  sb_append_char(&val, '\f'); break;
                    case '"':  sb_append_char(&val, '"');  break;
                    case '\\': sb_append_char(&val, '\\'); break;
                    case '/':  sb_append_char(&val, '/');  break;
                    case 'u': {
                        if (!p[1] || !p[2] || !p[3] || !p[4]) { sb_free(&val); goto fail; }
                        char hex[5] = { p[1], p[2], p[3], p[4], 0 };
                        unsigned int code = (unsigned int)strtoul(hex, NULL, 16);
                        if (code < 0x80) {
                            sb_append_char(&val, (char)code);
                        } else if (code < 0x800) {
                            sb_append_char(&val, (char)(0xC0 | (code >> 6)));
                            sb_append_char(&val, (char)(0x80 | (code & 0x3F)));
                        } else {
                            sb_append_char(&val, (char)(0xE0 | (code >> 12)));
                            sb_append_char(&val, (char)(0x80 | ((code >> 6) & 0x3F)));
                            sb_append_char(&val, (char)(0x80 | (code & 0x3F)));
                        }
                        p += 4;
                        break;
                    }
                    default: sb_append_char(&val, *p); break;
                }
                if (*p) p++;
            } else {
                sb_append_char(&val, *p++);
            }
        }
        if (*p != '"') { sb_free(&val); goto fail; }
        p++;
        sb_ensure(&val, 1);
        val.data[val.len] = '\0';

        if (strcmp(key, "role") == 0) {
            *out_role = val.data;
        } else if (strcmp(key, "content") == 0) {
            *out_content = val.data;
        } else {
            sb_free(&val);
        }
    }

    if (!*out_role || !*out_content) goto fail;
    return 0;

fail:
    free(*out_role); *out_role = NULL;
    free(*out_content); *out_content = NULL;
    return -1;
}

typedef struct {
    char  *path;
    char  *id;
    time_t mtime;
    char  *preview;
} SessionEntry;

static int session_compare_desc(const void *a, const void *b) {
    const SessionEntry *sa = a, *sb = b;
    if (sb->mtime > sa->mtime) return 1;
    if (sb->mtime < sa->mtime) return -1;
    return 0;
}

static int session_scan(const char *dir, SessionEntry **out) {
    DIR *d = opendir(dir);
    if (!d) { *out = NULL; return 0; }

    SessionEntry *list = NULL;
    int count = 0, cap = 0;
    struct dirent *ent;
    while ((ent = readdir(d))) {
        const char *name = ent->d_name;
        size_t nlen = strlen(name);
        if (nlen < 7 || strcmp(name + nlen - 6, ".jsonl") != 0) continue;

        if (count >= cap) {
            cap = cap ? cap * 2 : 8;
            list = realloc(list, cap * sizeof(SessionEntry));
        }
        SessionEntry *e = &list[count++];
        size_t plen = strlen(dir) + nlen + 2;
        e->path = malloc(plen);
        snprintf(e->path, plen, "%s/%s", dir, name);
        e->id = malloc(nlen - 5);
        memcpy(e->id, name, nlen - 6);
        e->id[nlen - 6] = '\0';
        e->preview = NULL;
        e->mtime = 0;

        struct stat st;
        if (stat(e->path, &st) == 0) e->mtime = st.st_mtime;

        FILE *f = fopen(e->path, "r");
        if (f) {
            char *line = NULL;
            size_t line_cap = 0;
            while (getline(&line, &line_cap, f) != -1) {
                char *role = NULL, *content = NULL;
                if (parse_session_line(line, &role, &content) == 0) {
                    if (strcmp(role, "user") == 0) {
                        size_t clen = strlen(content);
                        size_t show = clen > 60 ? 60 : clen;
                        e->preview = malloc(show + 4);
                        memcpy(e->preview, content, show);
                        if (clen > 60) { strcpy(e->preview + show, "..."); }
                        else { e->preview[show] = '\0'; }
                        /* strip newlines for display */
                        for (char *q = e->preview; *q; q++)
                            if (*q == '\n' || *q == '\r' || *q == '\t') *q = ' ';
                        free(role); free(content);
                        break;
                    }
                    free(role); free(content);
                }
            }
            free(line);
            fclose(f);
        }
        if (!e->preview) e->preview = strdup("(empty)");
    }
    closedir(d);

    qsort(list, count, sizeof(SessionEntry), session_compare_desc);
    *out = list;
    return count;
}

static void session_entries_free(SessionEntry *list, int count) {
    for (int i = 0; i < count; i++) {
        free(list[i].path);
        free(list[i].id);
        free(list[i].preview);
    }
    free(list);
}

/* Returns malloc'd path of a session to load, or NULL to start a new one. */
char *session_picker(const char *dir) {
    SessionEntry *list = NULL;
    int n = session_scan(dir, &list);
    if (n == 0) return NULL;

    printf("\033[1mPrevious sessions in this directory:\033[0m\n");
    int max_show = n > 9 ? 9 : n;
    for (int i = 0; i < max_show; i++) {
        struct tm *t = localtime(&list[i].mtime);
        char ts[32];
        strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M", t);
        printf("  \033[33m%d\033[0m) %s  \033[90m%s\033[0m\n",
               i + 1, ts, list[i].preview);
    }
    printf("  \033[33m0\033[0m) Start new session\n");
    printf("\nChoice [0]: ");
    fflush(stdout);

    char buf[16];
    char *result = NULL;
    if (fgets(buf, sizeof(buf), stdin)) {
        int choice = atoi(buf);
        if (choice >= 1 && choice <= max_show)
            result = strdup(list[choice - 1].path);
    }
    session_entries_free(list, n);
    return result;
}

/* Replays a session file into the messages array. Skips system records.
 * Walks newest→oldest, accumulating messages whose total content fits in
 * 30% of n_ctx tokens (estimated 4 chars per token). Caller owns the message
 * contents (each is a fresh strdup'd buffer). */
void session_load_into(
    const char *path,
    struct llama_chat_message **messages,
    size_t *msg_count,
    size_t *msg_cap,
    int n_ctx)
{
    FILE *f = fopen(path, "r");
    if (!f) return;

    char **roles = NULL;
    char **contents = NULL;
    int count = 0, cap = 0;

    char *line = NULL;
    size_t line_cap = 0;
    while (getline(&line, &line_cap, f) != -1) {
        char *role = NULL, *content = NULL;
        if (parse_session_line(line, &role, &content) != 0) continue;
        if (strcmp(role, "system") == 0) {
            free(role); free(content);
            continue;
        }
        if (count >= cap) {
            cap = cap ? cap * 2 : 32;
            roles    = realloc(roles,    cap * sizeof(char *));
            contents = realloc(contents, cap * sizeof(char *));
        }
        roles[count] = role;
        contents[count] = content;
        count++;
    }
    free(line);
    fclose(f);

    /* 30% of n_ctx in tokens, ~4 chars per token. */
    size_t budget_chars = (size_t)((double)n_ctx * 4.0 * 0.30);
    size_t total = 0;
    int start = count;
    for (int i = count - 1; i >= 0; i--) {
        size_t clen = strlen(contents[i]);
        if (total + clen > budget_chars && start < count) break;
        total += clen;
        start = i;
    }

    int loaded = 0;
    for (int i = start; i < count; i++) {
        const char *role_lit;
        if (strcmp(roles[i], "user") == 0)           role_lit = "user";
        else if (strcmp(roles[i], "assistant") == 0) role_lit = "assistant";
        else { free(roles[i]); free(contents[i]); continue; }

        if (*msg_count >= *msg_cap) {
            *msg_cap = *msg_cap ? *msg_cap * 2 : 16;
            *messages = realloc(*messages,
                *msg_cap * sizeof(struct llama_chat_message));
        }
        (*messages)[*msg_count].role = role_lit;
        (*messages)[*msg_count].content = contents[i];
        (*msg_count)++;
        loaded++;
        free(roles[i]);
        contents[i] = NULL;
    }

    for (int i = 0; i < start; i++) {
        free(roles[i]);
        free(contents[i]);
    }
    free(roles);
    free(contents);

    printf("\033[90m[Resumed: %d of %d messages, ~%zu chars (budget %zu)]\033[0m\n",
           loaded, count, total, budget_chars);
    fflush(stdout);
}

FILE *session_open_new(const char *dir, char **out_path) {
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char id[32];
    strftime(id, sizeof(id), "%Y%m%d-%H%M%S", t);
    size_t plen = strlen(dir) + strlen(id) + 16;
    char *path = malloc(plen);
    snprintf(path, plen, "%s/%s.jsonl", dir, id);
    FILE *fp = fopen(path, "w");
    if (out_path) *out_path = path;
    else free(path);
    return fp;
}
