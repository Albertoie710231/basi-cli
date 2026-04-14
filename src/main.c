#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <termios.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <poll.h>
#include <dirent.h>
#include <libgen.h>

#include "llama.h"

#define MAX_TOKENS          8192
#define CONTEXT_SIZE        32768
#define MAX_FILE_TOKENS     2000
#define MAX_TOOL_RESULT_SZ  16000  /* max chars in a tool result (~4000 tokens) */
#define MAX_HISTORY         100
#define FORMATTED_BUF_SZ   (MAX_TOKENS * 4)

/* ── System prompt ─────────────────────────────────────────────────── */

static const char *SYSTEM_PROMPT_FMT =
    "You are BASI, a helpful AI assistant with access to file and web tools.\n"
    "Today's date is %s.\n"
    "Your training data may be outdated. Always use the search tool for current events, recent products, or anything time-sensitive.\n"
    "\n"
    "When you need to read files, search the web, or fetch URLs, use these tools by wrapping commands in <tool> tags:\n"
    "\n"
    "FILE TOOLS:\n"
    "- read <file> : Read entire file (only for small files <2000 tokens)\n"
    "- head -n <N> <file> : Read first N lines\n"
    "- grep <pattern> <file> : Search for pattern (use quotes for multi-word)\n"
    "- grep -n <pattern> <file> : Search with line numbers\n"
    "- grep -C <N> <pattern> <file> : Search with N lines of context\n"
    "- wc <file> : Count lines, words, characters\n"
    "\n"
    "WEB TOOL:\n"
    "- webfetch \"search query\" \"grep regex\" : Takes exactly 2 quoted arguments. First is a web search query. Second is a regex pattern (use | for OR). Fetches top 5 results in parallel and extracts lines matching the regex.\n"
    "\n"
    "Examples:\n"
    "<tool>head -n 50 /path/to/file.txt</tool>\n"
    "<tool>grep -n \"function main\" src/main.c</tool>\n"
    "<tool>webfetch \"latest google pixel phone specs\" \"pixel|specs|price|release|camera\"</tool>\n"
    "<tool>webfetch \"rust async tutorial\" \"async|await|Future|tokio\"</tool>\n"
    "\n"
    "Tool results will appear in <tool_result> tags.\n"
    "For large files, first use 'wc' to check size, then 'head' or 'grep' to read relevant parts.\n"
    "CRITICAL RULES:\n"
    "1. After receiving tool results, you MUST answer the user immediately. Do not call another tool unless the results were completely empty.\n"
    "2. Base your answers on the actual content returned by tools. Never assume details.\n"
    "3. One webfetch call is almost always enough. Answer with what you have.\n"
    "\n"
    "Always be helpful, concise, and accurate.";

/* ── Spinner frames ────────────────────────────────────────────────── */

static const char *spinner_frames[] = {
    "\xe2\xa0\x8b", "\xe2\xa0\x99", "\xe2\xa0\xb9", "\xe2\xa0\xb8",
    "\xe2\xa0\xbc", "\xe2\xa0\xb4", "\xe2\xa0\xa6", "\xe2\xa0\xa7",
    "\xe2\xa0\x87", "\xe2\xa0\x8f"
};
#define SPINNER_COUNT 10

/* ── Dynamic string buffer ─────────────────────────────────────────── */

typedef struct {
    char  *data;
    size_t len;
    size_t cap;
} StringBuf;

static void sb_init(StringBuf *sb) {
    sb->data = NULL;
    sb->len  = 0;
    sb->cap  = 0;
}

static void sb_free(StringBuf *sb) {
    free(sb->data);
    sb->data = NULL;
    sb->len = sb->cap = 0;
}

static void sb_ensure(StringBuf *sb, size_t extra) {
    size_t need = sb->len + extra;
    if (need <= sb->cap) return;
    size_t newcap = sb->cap ? sb->cap * 2 : 256;
    while (newcap < need) newcap *= 2;
    sb->data = realloc(sb->data, newcap);
    sb->cap = newcap;
}

static void sb_append(StringBuf *sb, const char *s, size_t n) {
    sb_ensure(sb, n);
    memcpy(sb->data + sb->len, s, n);
    sb->len += n;
}

static void sb_append_str(StringBuf *sb, const char *s) {
    sb_append(sb, s, strlen(s));
}

static void sb_append_char(StringBuf *sb, char c) {
    sb_ensure(sb, 1);
    sb->data[sb->len++] = c;
}

static char *sb_to_str(StringBuf *sb) {
    sb_ensure(sb, 1);
    sb->data[sb->len] = '\0';
    return sb->data;
}

static void sb_clear(StringBuf *sb) {
    sb->len = 0;
}

/* ── Signal handling ───────────────────────────────────────────────── */

static volatile sig_atomic_t generation_interrupted = 0;
static volatile sig_atomic_t show_thinking = 0; /* toggled by Ctrl+T */

static void sigint_handler(int sig) {
    (void)sig;
    generation_interrupted = 1;
}

static void setup_sigint_handler(void) {
    struct sigaction sa = {0};
    sa.sa_handler = sigint_handler;
    sa.sa_flags = SA_RESTART;
    sigaction(SIGINT, &sa, NULL);
}

static void reset_sigint_handler(void) {
    struct sigaction sa = {0};
    sa.sa_handler = SIG_DFL;
    sigaction(SIGINT, &sa, NULL);
}

/* ── Command history ───────────────────────────────────────────────── */

static char *history[MAX_HISTORY];
static int   history_count = 0;

static void history_add(const char *line) {
    if (!line || !line[0]) return;
    /* skip duplicate of last entry */
    if (history_count > 0 && strcmp(history[history_count - 1], line) == 0)
        return;
    if (history_count >= MAX_HISTORY) {
        free(history[0]);
        memmove(history, history + 1, (MAX_HISTORY - 1) * sizeof(char *));
        history_count--;
    }
    history[history_count++] = strdup(line);
}

static void history_free_all(void) {
    for (int i = 0; i < history_count; i++)
        free(history[i]);
    history_count = 0;
}

/* ── Terminal raw mode ─────────────────────────────────────────────── */

static struct termios orig_termios;
static bool raw_mode_enabled = false;

static bool enable_raw_mode(void) {
    if (tcgetattr(STDIN_FILENO, &orig_termios) < 0) return false;
    struct termios raw = orig_termios;
    raw.c_lflag &= ~(ICANON | ECHO | ISIG);
    raw.c_cc[VMIN]  = 1;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) < 0) return false;
    raw_mode_enabled = true;
    return true;
}

static void disable_raw_mode(void) {
    if (raw_mode_enabled) {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
        raw_mode_enabled = false;
    }
}

/* ── Line editor ───────────────────────────────────────────────────── */

/* Returns malloc'd string, or NULL on EOF. Empty string on Ctrl-C. */
static char *read_line(const char *prompt) {
    bool raw = enable_raw_mode();

    printf("%s", prompt);
    fflush(stdout);

    StringBuf line;
    sb_init(&line);
    size_t cursor = 0;
    int hist_idx = history_count;

    StringBuf saved;
    sb_init(&saved);

    while (1) {
        unsigned char ch;
        ssize_t n = read(STDIN_FILENO, &ch, 1);
        if (n <= 0) {
            /* EOF */
            if (line.len == 0) {
                sb_free(&line);
                sb_free(&saved);
                if (raw) disable_raw_mode();
                return NULL;
            }
            break;
        }

        if (ch == '\n' || ch == '\r') {
            printf("\n");
            fflush(stdout);
            break;
        } else if (ch == 27) {
            /* escape sequence */
            unsigned char seq[2];
            if (read(STDIN_FILENO, seq, 2) < 2) continue;
            if (seq[0] == '[') {
                switch (seq[1]) {
                case 'A': /* Up - history prev */
                    if (history_count > 0 && hist_idx > 0) {
                        if (hist_idx == history_count) {
                            sb_clear(&saved);
                            sb_append(&saved, line.data, line.len);
                        }
                        hist_idx--;
                        if (cursor > 0) printf("\033[%zuD", cursor);
                        printf("\033[K");
                        sb_clear(&line);
                        sb_append_str(&line, history[hist_idx]);
                        cursor = line.len;
                        fwrite(line.data, 1, line.len, stdout);
                        fflush(stdout);
                    }
                    break;
                case 'B': /* Down - history next */
                    if (hist_idx < history_count) {
                        hist_idx++;
                        if (cursor > 0) printf("\033[%zuD", cursor);
                        printf("\033[K");
                        sb_clear(&line);
                        if (hist_idx < history_count) {
                            sb_append_str(&line, history[hist_idx]);
                        } else {
                            sb_append(&line, saved.data, saved.len);
                        }
                        cursor = line.len;
                        fwrite(line.data, 1, line.len, stdout);
                        fflush(stdout);
                    }
                    break;
                case 'C': /* Right */
                    if (cursor < line.len) {
                        cursor++;
                        printf("\033[C");
                        fflush(stdout);
                    }
                    break;
                case 'D': /* Left */
                    if (cursor > 0) {
                        cursor--;
                        printf("\033[D");
                        fflush(stdout);
                    }
                    break;
                case 'H': /* Home */
                    if (cursor > 0) {
                        printf("\033[%zuD", cursor);
                        fflush(stdout);
                        cursor = 0;
                    }
                    break;
                case 'F': /* End */
                    if (cursor < line.len) {
                        printf("\033[%zuC", line.len - cursor);
                        fflush(stdout);
                        cursor = line.len;
                    }
                    break;
                case '3': { /* Delete key: [3~ */
                    unsigned char tilde;
                    read(STDIN_FILENO, &tilde, 1);
                    if (cursor < line.len) {
                        memmove(line.data + cursor, line.data + cursor + 1,
                                line.len - cursor - 1);
                        line.len--;
                        printf("\033[s");
                        fwrite(line.data + cursor, 1, line.len - cursor, stdout);
                        printf(" \033[u");
                        fflush(stdout);
                    }
                    break;
                }
                default:
                    break;
                }
            }
        } else if (ch == 127 || ch == 8) {
            /* Backspace */
            if (cursor > 0) {
                cursor--;
                memmove(line.data + cursor, line.data + cursor + 1,
                        line.len - cursor - 1);
                line.len--;
                printf("\033[D\033[s");
                fwrite(line.data + cursor, 1, line.len - cursor, stdout);
                printf(" \033[u");
                fflush(stdout);
            }
        } else if (ch == 3) {
            /* Ctrl-C */
            printf("^C\n");
            fflush(stdout);
            sb_free(&line);
            sb_free(&saved);
            if (raw) disable_raw_mode();
            return strdup("");
        } else if (ch == 4) {
            /* Ctrl-D */
            if (line.len == 0) {
                sb_free(&line);
                sb_free(&saved);
                if (raw) disable_raw_mode();
                return NULL;
            }
        } else if (ch >= 32) {
            /* Printable */
            sb_ensure(&line, 1);
            if (cursor < line.len) {
                memmove(line.data + cursor + 1, line.data + cursor,
                        line.len - cursor);
            }
            line.data[cursor] = ch;
            line.len++;
            cursor++;
            if (cursor == line.len) {
                putchar(ch);
                fflush(stdout);
            } else {
                printf("\033[s");
                putchar(ch);
                fwrite(line.data + cursor, 1, line.len - cursor, stdout);
                printf("\033[u\033[C");
                fflush(stdout);
            }
        }
    }

    if (raw) disable_raw_mode();

    /* null-terminate and return */
    char *result = malloc(line.len + 1);
    if (line.len > 0) memcpy(result, line.data, line.len);
    result[line.len] = '\0';

    history_add(result);

    sb_free(&line);
    sb_free(&saved);
    return result;
}

/* ── URL encode/decode ─────────────────────────────────────────────── */

static char *url_encode(const char *input) {
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

static char *url_decode(const char *input) {
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

/* ── Count lines in file ───────────────────────────────────────────── */

static size_t count_lines(FILE *f) {
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

/* ── Run a shell command and capture output ────────────────────────── */

static char *run_command(const char *cmd, size_t max_output) {
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

/* ── Webfetch: search + parallel fetch + grep ──────────────────────── */

#define WEBFETCH_MAX_RESULTS   5
#define WEBFETCH_MAX_CHARS  2500  /* max chars of extracted content per page */
#define WEBFETCH_MIN_LINE_LEN 30  /* skip short lines (nav, buttons, junk) */

static bool debug_mode = false;

/*
 * Fetch a URL, strip HTML, then use awk to extract paragraphs around matches.
 *
 * The awk script:
 * - Skips short lines (<MIN_LINE_LEN chars) to filter nav/button junk
 * - Stores lines in a ring buffer of 2 (prev context)
 * - When a match is found, prints prev + match + next line (paragraph)
 * - Deduplicates: won't print the same line twice
 * - Stops after MAX_CHARS total output
 * - Separates match groups with "---"
 */
static char *fetch_and_extract(const char *url, const char *pattern) {
    /* Single pipeline: fetch → strip HTML → filter junk → awk extract.
       Awk handles both small and large pages:
       - Counts total usable chars in first pass through lines
       - If page is small (<=5000 usable chars), outputs everything
       - If large, extracts paragraphs around pattern matches
       - Hard cap via head -c at the end */
    char cmd[16384];
    snprintf(cmd, sizeof(cmd),
        "curl -sL -m 10 -A 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36' "
        "'%s' 2>/dev/null "
        "| w3m -dump -T text/html -cols 120 2>/dev/null "
        "| awk -v pat='%s' -v maxchars=%d -v minlen=%d -v dbg=%d '"
        "{"
        "  gsub(/^[[:space:]]+|[[:space:]]+$/, \"\");"
        "  if (length($0) < minlen) next;"
        "  lines[++n] = $0; sz += length($0) + 1"
        "}"
        "END {"
        "  IGNORECASE=1;"
        "  if (sz <= 5000) {"
        "    for (i=1; i<=n; i++) print lines[i];"
        "    if (dbg) printf \"[DEBUG] %%d bytes (small page, full output)\\n\", sz > \"/dev/stderr\";"
        "  } else {"
        "    total=0; lastprinted=-10; matches=0;"
        "    for (i=1; i<=n; i++) {"
        "      if (total >= maxchars) break;"
        "      if (lines[i] ~ pat) {"
        "        matches++;"
        "        if (i - lastprinted > 3 && total > 0) { print \"---\"; total+=4 }"
        "        if (total >= maxchars) break;"
        "        if (i - lastprinted > 2 && i>=3) { print lines[i-2]; total+=length(lines[i-2])+1 }"
        "        if (total >= maxchars) break;"
        "        if (i - lastprinted > 1 && i>=2) { print lines[i-1]; total+=length(lines[i-1])+1 }"
        "        if (total >= maxchars) break;"
        "        print lines[i]; total+=length(lines[i])+1;"
        "        lastprinted=i;"
        "        if (i+1<=n && total<maxchars) { print lines[i+1]; total+=length(lines[i+1])+1; lastprinted=i+1 }"
        "        if (i+2<=n && total<maxchars) { print lines[i+2]; total+=length(lines[i+2])+1; lastprinted=i+2 }"
        "      }"
        "    }"
        "    if (dbg) printf \"[DEBUG] %%d bytes page, %%d matches, %%d bytes output\\n\", sz, matches, total > \"/dev/stderr\";"
        "  }"
        "}' | head -c %d",
        url, pattern, WEBFETCH_MAX_CHARS, WEBFETCH_MIN_LINE_LEN, debug_mode ? 1 : 0,
        WEBFETCH_MAX_CHARS);

    char *result = run_command(cmd, WEBFETCH_MAX_CHARS + 256);
    /* Hard truncate if still too long */
    if (result && strlen(result) > (size_t)WEBFETCH_MAX_CHARS) {
        result[WEBFETCH_MAX_CHARS] = '\0';
    }
    return result;
}

/* webfetch <search_query> <grep_query>
   1. Search DDG for search_query → 5 URLs
   2. Fetch all 5 in parallel with mini_browser
   3. Grep each for grep_query
   4. Return structured results */
static char *execute_webfetch(const char *search_query, const char *grep_query) {
    char *encoded = url_encode(search_query);
    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
        "/home/alberto/Documentos/MINI_BROWSER/build/mini_browser --headless --timeout 30000 "
        "'https://lite.duckduckgo.com/lite/?q=%s' 2>/dev/null "
        "| grep -oP '(?<=class=\"result-link\" href=\")[^\"]+' "
        "| grep -v 'ad_domain' | grep -v 'ad_provider' "
        "| grep -v 'duckduckgo.com/duckduckgo-help-pages' | head -15",
        encoded);
    free(encoded);

    char *raw = run_command(cmd, 256 * 1024);
    if (!raw || !raw[0]) {
        free(raw);
        return strdup("No search results found.");
    }

    /* Parse DDG redirect URLs → collect up to 5 */
    char *urls[WEBFETCH_MAX_RESULTS];
    int url_count = 0;

    char *raw_copy = strdup(raw);
    char *line = strtok(raw_copy, "\n");
    while (line && url_count < WEBFETCH_MAX_RESULTS) {
        char *uddg = strstr(line, "uddg=");
        if (uddg) {
            uddg += 5;
            char *end = strchr(uddg, '&');
            if (!end) end = uddg + strlen(uddg);
            char saved = *end;
            *end = '\0';
            urls[url_count] = url_decode(uddg);
            *end = saved;
            url_count++;
        }
        line = strtok(NULL, "\n");
    }
    free(raw_copy);
    free(raw);

    if (url_count == 0)
        return strdup("No search results found.");

    printf("\033[90m[Fetching %d results in parallel, grepping for: %s]\033[0m\n",
           url_count, grep_query);
    fflush(stdout);

    /* Fork parallel fetchers */
    int pipes[WEBFETCH_MAX_RESULTS][2];
    pid_t pids[WEBFETCH_MAX_RESULTS];

    for (int i = 0; i < url_count; i++) {
        pipe(pipes[i]);
        pids[i] = fork();
        if (pids[i] == 0) {
            close(pipes[i][0]);
            char *content = fetch_and_extract(urls[i], grep_query);
            if (content && content[0]) {
                write(pipes[i][1], content, strlen(content));
            }
            close(pipes[i][1]);
            free(content);
            _exit(0);
        } else {
            close(pipes[i][1]);
        }
    }

    /* Parent: collect results from all children */
    StringBuf results;
    sb_init(&results);

    for (int i = 0; i < url_count; i++) {
        StringBuf child_out;
        sb_init(&child_out);
        char buf[4096];
        ssize_t n;
        while ((n = read(pipes[i][0], buf, sizeof(buf))) > 0) {
            sb_append(&child_out, buf, n);
        }
        close(pipes[i][0]);
        waitpid(pids[i], NULL, 0);

        char header[1024];
        snprintf(header, sizeof(header), "\n--- Result %d: %s ---\n", i + 1, urls[i]);
        sb_append_str(&results, header);

        if (child_out.len > 0) {
            size_t cap = child_out.len < (size_t)WEBFETCH_MAX_CHARS
                       ? child_out.len : (size_t)WEBFETCH_MAX_CHARS;
            sb_append(&results, child_out.data, cap);
        } else {
            sb_append_str(&results, "(no relevant content found)\n");
        }
        sb_free(&child_out);
    }

    /* Cleanup */
    for (int i = 0; i < url_count; i++)
        free(urls[i]);

    if (results.len == 0) {
        sb_free(&results);
        return strdup("No search results found.");
    }
    return sb_to_str(&results);
}

/* ── Tokenize command string (respecting quotes) ───────────────────── */

typedef struct {
    char **args;
    int    count;
} ArgList;

static ArgList tokenize_command(const char *cmd) {
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

static void arglist_free(ArgList *al) {
    for (int i = 0; i < al->count; i++)
        free(al->args[i]);
    free(al->args);
    al->args = NULL;
    al->count = 0;
}

/* ── Execute tool command ──────────────────────────────────────────── */

static char *execute_tool(const char *command) {
    /* trim whitespace */
    while (*command == ' ' || *command == '\t' || *command == '\n') command++;
    if (!*command) return strdup("Error: Empty command");

    ArgList al = tokenize_command(command);
    if (al.count == 0) {
        arglist_free(&al);
        return strdup("Error: No command specified");
    }

    const char *cmd = al.args[0];

    /* Whitelist check */
    static const char *allowed[] = {
        "read", "head", "tail", "grep", "wc", "cat", "webfetch", NULL
    };
    bool is_allowed = false;
    for (const char **a = allowed; *a; a++) {
        if (strcmp(cmd, *a) == 0) { is_allowed = true; break; }
    }
    if (!is_allowed) {
        char *msg = malloc(256);
        snprintf(msg, 256,
            "Error: Command '%s' not allowed. Use: read, head, tail, grep, wc, webfetch", cmd);
        arglist_free(&al);
        return msg;
    }

    /* Handle 'read' → file read with size check */
    if (strcmp(cmd, "read") == 0) {
        if (al.count < 2) {
            arglist_free(&al);
            return strdup("Error: read requires a file path");
        }
        const char *filepath = al.args[1];
        FILE *f = fopen(filepath, "r");
        if (!f) {
            char *msg = malloc(512);
            snprintf(msg, 512, "Error: Cannot open file '%s'", filepath);
            arglist_free(&al);
            return msg;
        }
        struct stat st;
        fstat(fileno(f), &st);
        size_t estimated_tokens = st.st_size / 4;
        if (estimated_tokens > MAX_FILE_TOKENS) {
            size_t lines = count_lines(f);
            char *msg = malloc(512);
            snprintf(msg, 512,
                "Error: File too large (~%zu tokens, max %d). Use 'head', 'tail', or 'grep' to read in chunks.\n"
                "File has %ld bytes, %zu lines (use 'wc %s' for exact count)",
                estimated_tokens, MAX_FILE_TOKENS, (long)st.st_size, lines, filepath);
            fclose(f);
            arglist_free(&al);
            return msg;
        }
        char *content = malloc(st.st_size + 1);
        size_t nread = fread(content, 1, st.st_size, f);
        content[nread] = '\0';
        fclose(f);
        arglist_free(&al);
        return content;
    }

    /* Handle 'webfetch' — search + parallel fetch + grep */
    if (strcmp(cmd, "webfetch") == 0) {
        if (al.count < 3) {
            arglist_free(&al);
            return strdup("Error: webfetch requires two arguments:\n"
                          "  webfetch \"search query\" \"grep pattern\"\n"
                          "Example: webfetch \"google pixel 10 specs\" \"pixel 10|price|specs|release\"");
        }
        char *result = execute_webfetch(al.args[1], al.args[2]);
        arglist_free(&al);
        return result;
    }

    /* Build shell command for head/tail/grep/wc/cat */
    StringBuf shell_cmd;
    sb_init(&shell_cmd);

    const char *actual = strcmp(cmd, "read") == 0 ? "cat" : cmd;
    sb_append_str(&shell_cmd, actual);

    for (int i = 1; i < al.count; i++) {
        sb_append_char(&shell_cmd, ' ');
        /* Quote arguments that need it */
        if (strpbrk(al.args[i], " \t\"'$`\\")) {
            sb_append_char(&shell_cmd, '\'');
            for (const char *c = al.args[i]; *c; c++) {
                if (*c == '\'') {
                    sb_append_str(&shell_cmd, "'\"'\"'");
                } else {
                    sb_append_char(&shell_cmd, *c);
                }
            }
            sb_append_char(&shell_cmd, '\'');
        } else {
            sb_append_str(&shell_cmd, al.args[i]);
        }
    }

    char *result = run_command(sb_to_str(&shell_cmd), 512 * 1024);
    sb_free(&shell_cmd);
    arglist_free(&al);
    return result;
}

/* ── Extract <tool>...</tool> from text ────────────────────────────── */

static const char *extract_tool_call(const char *text, size_t *out_len) {
    const char *start = strstr(text, "<tool>");
    if (!start) return NULL;
    start += 6; /* strlen("<tool>") */
    const char *end = strstr(start, "</tool>");
    if (!end) return NULL;
    *out_len = end - start;
    return start;
}

/* ── Thinking animation ────────────────────────────────────────────── */

static void draw_thinking_box(size_t frame) {
    const char *spinner = spinner_frames[frame % SPINNER_COUNT];
    printf("\r\033[K\033[90m\xe2\x94\x8c\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x90\033[0m\r\n");
    printf("\033[90m\xe2\x94\x82\033[0m \033[36m%s thinking...\033[0m   "
           "\033[90m\xe2\x94\x82\033[0m\r\n", spinner);
    printf("\033[90m\xe2\x94\x94\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x98\033[0m");
    printf("\033[2A");
    fflush(stdout);
}

static void clear_thinking_box(void) {
    printf("\r\033[K\n\033[K\n\033[K\033[2A\r");
    fflush(stdout);
}

/* ── UTF-8 helpers ─────────────────────────────────────────────────── */

static size_t utf8_seq_len(unsigned char b) {
    if ((b & 0x80) == 0)    return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 1;
}

/* ── Thinking state machine ────────────────────────────────────────── */

typedef enum {
    STATE_NORMAL,
    STATE_MAYBE_OPEN,
    STATE_THINKING,
    STATE_MAYBE_CLOSE,
} ThinkingState;

/* ── Generate response ─────────────────────────────────────────────── */

typedef struct {
    char  *text;
    size_t prompt_tokens;
    size_t gen_tokens;
    double prompt_time_s;
    double gen_time_s;
} GenerateResult;

static double time_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static GenerateResult generate(
    struct llama_context *ctx,
    const struct llama_vocab *vocab,
    struct llama_sampler *smpl,
    const char *prompt,
    size_t prompt_len)
{
    GenerateResult res = { NULL, 0, 0, 0.0, 0.0 };
    StringBuf response;
    sb_init(&response);

    /* Check if first generation */
    llama_memory_t memory = llama_get_memory(ctx);
    bool is_first = (llama_memory_seq_pos_max(memory, 0) == -1);

    /* Tokenize */
    int n_prompt_tokens = -llama_tokenize(vocab, prompt, (int)prompt_len,
                                           NULL, 0, is_first, true);
    if (n_prompt_tokens <= 0) {
        res.text = strdup("[Tokenization failed]");
        return res;
    }

    llama_token *tokens = malloc(n_prompt_tokens * sizeof(llama_token));
    llama_tokenize(vocab, prompt, (int)prompt_len,
                   tokens, n_prompt_tokens, is_first, true);

    res.prompt_tokens = n_prompt_tokens;

    /* Create batch */
    struct llama_batch batch = llama_batch_get_one(tokens, n_prompt_tokens);
    bool is_prompt_phase = true;
    double timer_start = time_now();

    /* Thinking state */
    ThinkingState state = STATE_NORMAL;
    char tag_buf[16];
    size_t tag_len = 0;
    size_t spinner_frame = 0;
    double last_spinner = 0;
    bool thinking_box_shown = false;

    /* UTF-8 buffer */
    unsigned char utf8_buf[4];
    size_t utf8_len = 0;

    /* Generation loop */
    while (1) {
        /* Check context space */
        uint32_t n_ctx = llama_n_ctx(ctx);
        uint32_t n_ctx_used = (uint32_t)(llama_memory_seq_pos_max(memory, 0) + 1);
        if (n_ctx_used + (uint32_t)batch.n_tokens > n_ctx) {
            if (thinking_box_shown) { if (show_thinking) { printf("\033[0m\n"); } else { clear_thinking_box(); } thinking_box_shown = false; }
            printf("\n[Context limit reached]\n");
            fflush(stdout);
            break;
        }

        if (llama_decode(ctx, batch) != 0) {
            if (thinking_box_shown) { if (show_thinking) { printf("\033[0m\n"); } else { clear_thinking_box(); } thinking_box_shown = false; }
            printf("\n[Decode error]\n");
            fflush(stdout);
            break;
        }

        if (is_prompt_phase) {
            res.prompt_time_s = time_now() - timer_start;
            timer_start = time_now();
            is_prompt_phase = false;
        }

        llama_token new_token = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) {
            res.gen_time_s = time_now() - timer_start;
            if (thinking_box_shown) { if (show_thinking) { printf("\033[0m\n"); } else { clear_thinking_box(); } thinking_box_shown = false; }
            break;
        }

        if (generation_interrupted) {
            res.gen_time_s = time_now() - timer_start;
            if (thinking_box_shown) { if (show_thinking) { printf("\033[0m\n"); } else { clear_thinking_box(); } thinking_box_shown = false; }
            printf("\n\033[90m[interrupted]\033[0m");
            fflush(stdout);
            break;
        }

        /* Check for Ctrl+T (toggle thinking display) via non-blocking read */
        {
            struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN };
            if (poll(&pfd, 1, 0) > 0 && (pfd.revents & POLLIN)) {
                unsigned char key;
                if (read(STDIN_FILENO, &key, 1) == 1 && key == 0x14) { /* Ctrl+T */
                    show_thinking = !show_thinking;
                    if (state == STATE_THINKING || state == STATE_MAYBE_CLOSE) {
                        if (show_thinking) {
                            /* Switching from box to text */
                            if (thinking_box_shown) { if (show_thinking) { printf("\033[0m\n"); } else { clear_thinking_box(); } thinking_box_shown = false; }
                            printf("\033[90m[thinking] ");
                            fflush(stdout);
                        } else {
                            /* Switching from text to box */
                            printf("\033[0m\n");
                            draw_thinking_box(spinner_frame);
                            thinking_box_shown = true;
                            last_spinner = time_now();
                        }
                    }
                }
            }
        }

        res.gen_tokens++;

        /* Token to text */
        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (n < 0) continue;

        /* Process through thinking state machine */
        size_t piece_start = 0;
        for (int idx = 0; idx < n; idx++) {
            char ch = buf[idx];
            switch (state) {
            case STATE_NORMAL:
                if (ch == '<') {
                    /* Flush text before '<' */
                    if ((size_t)idx > piece_start) {
                        printf("\033[33m");
                        fwrite(buf + piece_start, 1, idx - piece_start, stdout);
                        fflush(stdout);
                        sb_append(&response, buf + piece_start, idx - piece_start);
                    }
                    state = STATE_MAYBE_OPEN;
                    tag_len = 0;
                    tag_buf[tag_len++] = ch;
                    piece_start = idx + 1;
                }
                break;

            case STATE_MAYBE_OPEN: {
                tag_buf[tag_len++] = ch;
                const char *target = "<think>";
                if (tag_len <= 7 && tag_buf[tag_len - 1] == target[tag_len - 1]) {
                    if (tag_len == 7) {
                        state = STATE_THINKING;
                        tag_len = 0;
                        piece_start = idx + 1;
                        if (show_thinking) {
                            printf("\033[90m[thinking] ");
                            fflush(stdout);
                        } else {
                            draw_thinking_box(spinner_frame);
                            last_spinner = time_now();
                        }
                        thinking_box_shown = true;
                    }
                } else {
                    /* Not <think>, flush tag buffer */
                    printf("\033[33m");
                    fwrite(tag_buf, 1, tag_len, stdout);
                    fflush(stdout);
                    sb_append(&response, tag_buf, tag_len);
                    state = STATE_NORMAL;
                    tag_len = 0;
                    piece_start = idx + 1;
                }
                break;
            }

            case STATE_THINKING: {
                if (ch == '<') {
                    state = STATE_MAYBE_CLOSE;
                    tag_len = 0;
                    tag_buf[tag_len++] = ch;
                } else if (show_thinking) {
                    printf("\033[90m%c", ch);
                    fflush(stdout);
                } else {
                    /* Spinner animation */
                    double now = time_now();
                    if (now - last_spinner > 0.08) {
                        spinner_frame++;
                        draw_thinking_box(spinner_frame);
                        last_spinner = now;
                    }
                }
                piece_start = idx + 1;
                break;
            }

            case STATE_MAYBE_CLOSE: {
                tag_buf[tag_len++] = ch;
                const char *target = "</think>";
                if (tag_len <= 8 && tag_buf[tag_len - 1] == target[tag_len - 1]) {
                    if (tag_len == 8) {
                        state = STATE_NORMAL;
                        tag_len = 0;
                        piece_start = idx + 1;
                        if (show_thinking) {
                            printf("\033[0m\n");
                        } else {
                            clear_thinking_box();
                        }
                        fflush(stdout);
                        thinking_box_shown = false;
                    }
                } else {
                    if (show_thinking) {
                        printf("\033[90m");
                        fwrite(tag_buf, 1, tag_len, stdout);
                        fflush(stdout);
                    }
                    state = STATE_THINKING;
                    tag_len = 0;
                    piece_start = idx + 1;
                }
                break;
            }
            }
        }

        /* Output remaining content in normal state with UTF-8 buffering */
        if (state == STATE_NORMAL && piece_start < (size_t)n) {
            const char *slice = buf + piece_start;
            size_t slice_len = n - piece_start;

            unsigned char combined[260];
            size_t combined_len = 0;

            /* Prepend buffered UTF-8 bytes */
            memcpy(combined, utf8_buf, utf8_len);
            combined_len = utf8_len;
            utf8_len = 0;

            memcpy(combined + combined_len, slice, slice_len);
            combined_len += slice_len;

            /* Find complete UTF-8 boundary */
            size_t output_end = 0;
            size_t pos = 0;
            while (pos < combined_len) {
                size_t slen = utf8_seq_len(combined[pos]);
                if (pos + slen <= combined_len) {
                    output_end = pos + slen;
                    pos += slen;
                } else {
                    break;
                }
            }

            if (output_end > 0) {
                printf("\033[33m");
                fwrite(combined, 1, output_end, stdout);
                fflush(stdout);
                sb_append(&response, (const char *)combined, output_end);
            }

            /* Buffer incomplete trailing bytes */
            if (output_end < combined_len) {
                memcpy(utf8_buf, combined + output_end, combined_len - output_end);
                utf8_len = combined_len - output_end;
            }
        }

        /* Next token */
        llama_token single = new_token;
        batch = llama_batch_get_one(&single, 1);
    }

    /* Flush remaining UTF-8 */
    if (utf8_len > 0) {
        printf("\033[33m");
        fwrite(utf8_buf, 1, utf8_len, stdout);
        sb_append(&response, (const char *)utf8_buf, utf8_len);
    }

    printf("\033[0m\n");
    fflush(stdout);

    free(tokens);
    res.text = sb_to_str(&response);
    return res;
}

/* ── Log callback (suppress non-errors) ────────────────────────────── */

static void log_callback(enum ggml_log_level level, const char *text, void *user_data) {
    (void)user_data;
    if (level >= GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "%s", text);
    }
}

/* ── Custom ChatML template ────────────────────────────────────────── */

/*
 * Format messages as ChatML. Same format for all models:
 *   <|im_start|>role\ncontent<|im_end|>\n
 *
 * If add_generation_prompt is true, appends <|im_start|>assistant\n
 * If buf is NULL, returns the required length without writing.
 */
static int apply_chatml(
    const struct llama_chat_message *msgs, size_t n_msgs,
    bool add_gen_prompt,
    char *buf, size_t buf_size)
{
    size_t total = 0;

    #define CHATML_WRITE(s, len) do { \
        if (buf && total + (len) < buf_size) \
            memcpy(buf + total, (s), (len)); \
        total += (len); \
    } while(0)
    #define CHATML_STR(s) CHATML_WRITE(s, strlen(s))

    for (size_t i = 0; i < n_msgs; i++) {
        CHATML_STR("<|im_start|>");
        CHATML_STR(msgs[i].role);
        CHATML_WRITE("\n", 1);
        if (msgs[i].content)
            CHATML_STR(msgs[i].content);
        CHATML_STR("<|im_end|>\n");
    }

    if (add_gen_prompt) {
        CHATML_STR("<|im_start|>assistant\n");
    }

    #undef CHATML_WRITE
    #undef CHATML_STR

    if (buf && total < buf_size) buf[total] = '\0';
    return (int)total;
}

/* ── Model picker ──────────────────────────────────────────────────── */

#define MODEL_DIRS_MAX 4
static const char *model_search_dirs[] = {
    NULL, /* filled at runtime: ~/.cache/llama.cpp */
    "/home/alberto/models",
    ".",
    NULL
};

/* Read max context length from GGUF metadata (returns 0 if not found) */
static int read_gguf_context_length(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    /* GGUF header: magic(4) + version(4) + tensor_count(8) + metadata_count(8) */
    uint32_t magic, version;
    uint64_t tensor_count, metadata_count;
    if (fread(&magic, 4, 1, f) != 1) { fclose(f); return 0; }
    if (fread(&version, 4, 1, f) != 1) { fclose(f); return 0; }
    if (fread(&tensor_count, 8, 1, f) != 1) { fclose(f); return 0; }
    if (fread(&metadata_count, 8, 1, f) != 1) { fclose(f); return 0; }

    int result = 0;
    for (uint64_t i = 0; i < metadata_count; i++) {
        /* key: len(8) + data */
        uint64_t key_len;
        if (fread(&key_len, 8, 1, f) != 1) break;
        if (key_len > 256) { /* skip huge keys */ fseek(f, key_len, SEEK_CUR); }
        char key[257] = {0};
        size_t rlen = key_len < 256 ? key_len : 256;
        if (fread(key, 1, rlen, f) != rlen) break;
        if (key_len > 256) key[256] = '\0';

        /* value type */
        uint32_t vtype;
        if (fread(&vtype, 4, 1, f) != 1) break;

        if (vtype == 4) { /* uint32 */
            uint32_t val;
            if (fread(&val, 4, 1, f) != 1) break;
            if (strstr(key, "context_length")) {
                result = (int)val;
                break;
            }
        } else if (vtype == 5 || vtype == 6) { fseek(f, 4, SEEK_CUR); }
        else if (vtype == 0 || vtype == 1 || vtype == 7) { fseek(f, 1, SEEK_CUR); }
        else if (vtype == 2 || vtype == 3) { fseek(f, 2, SEEK_CUR); }
        else if (vtype == 10 || vtype == 12) { fseek(f, 8, SEEK_CUR); }
        else if (vtype == 8) { /* string */
            uint64_t slen;
            if (fread(&slen, 8, 1, f) != 1) break;
            fseek(f, slen, SEEK_CUR);
        } else if (vtype == 9) { /* array */
            uint32_t atype;
            uint64_t alen;
            if (fread(&atype, 4, 1, f) != 1) break;
            if (fread(&alen, 8, 1, f) != 1) break;
            for (uint64_t a = 0; a < alen; a++) {
                if (atype == 8) {
                    uint64_t slen;
                    if (fread(&slen, 8, 1, f) != 1) break;
                    fseek(f, slen, SEEK_CUR);
                } else if (atype == 4 || atype == 5 || atype == 6) { fseek(f, 4, SEEK_CUR); }
                else if (atype == 0 || atype == 1 || atype == 7) { fseek(f, 1, SEEK_CUR); }
                else if (atype == 2 || atype == 3) { fseek(f, 2, SEEK_CUR); }
                else if (atype == 10 || atype == 12) { fseek(f, 8, SEEK_CUR); }
            }
        } else {
            break; /* unknown type, bail */
        }
    }
    fclose(f);
    return result;
}

/* Extract a display name from a gguf filename */
static const char *model_display_name(const char *path) {
    const char *base = strrchr(path, '/');
    return base ? base + 1 : path;
}

/* Get file size in MB */
static double file_size_mb(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) return st.st_size / (1024.0 * 1024.0);
    return 0;
}

/* Launch config */
typedef struct {
    char *model_path;  /* malloc'd, caller frees */
    int   gpu_layers;
    int   ctx_size;
    float temperature;
} LaunchConfig;

/* Settings values for ←/→ adjustment */
static const int gpu_layer_opts[]  = { 0, 10, 20, 30, 40, 50, 60, 80, 99 };
static const float temp_opts[]     = { 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f };
#define N_GPU_OPTS  (int)(sizeof(gpu_layer_opts)/sizeof(gpu_layer_opts[0]))
#define N_TEMP_OPTS (int)(sizeof(temp_opts)/sizeof(temp_opts[0]))
#define CTX_STEP    1024  /* slider step size */
#define CTX_MIN     1024
#define CTX_DEFAULT 32768

/*
 * Scan directories for .gguf files, show interactive menu with settings.
 * Returns filled LaunchConfig, or model_path=NULL on cancel.
 */
static LaunchConfig pick_model(void) {
    LaunchConfig cfg = { NULL, 99, CONTEXT_SIZE, 0.4f };

    /* Build search dirs */
    static char cache_dir[512];
    const char *home = getenv("HOME");
    if (home) {
        snprintf(cache_dir, sizeof(cache_dir), "%s/.cache/llama.cpp", home);
        model_search_dirs[0] = cache_dir;
    }

    /* Collect .gguf files */
    char **models = NULL;
    int count = 0, cap = 0;

    for (int d = 0; d < MODEL_DIRS_MAX && model_search_dirs[d]; d++) {
        DIR *dir = opendir(model_search_dirs[d]);
        if (!dir) continue;
        struct dirent *ent;
        while ((ent = readdir(dir)) != NULL) {
            size_t nlen = strlen(ent->d_name);
            if (nlen < 5) continue;
            if (strcmp(ent->d_name + nlen - 5, ".gguf") != 0) continue;
            if (strstr(ent->d_name, "mmproj") != NULL) continue;
            if (count >= cap) {
                cap = cap ? cap * 2 : 16;
                models = realloc(models, cap * sizeof(char *));
            }
            char fullpath[1024];
            snprintf(fullpath, sizeof(fullpath), "%s/%s",
                     model_search_dirs[d], ent->d_name);
            models[count++] = strdup(fullpath);
        }
        closedir(dir);
    }

    if (count == 0) {
        fprintf(stderr, "No .gguf models found in search directories.\n");
        free(models);
        return cfg;
    }

    /* Raw terminal */
    struct termios orig;
    bool raw = false;
    if (tcgetattr(STDIN_FILENO, &orig) == 0) {
        struct termios t = orig;
        t.c_lflag &= ~(ICANON | ECHO);
        t.c_cc[VMIN] = 1;
        t.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &t);
        raw = true;
    }

    /* Read max context for first model */
    int *model_max_ctx = calloc(count, sizeof(int));
    for (int i = 0; i < count; i++)
        model_max_ctx[i] = read_gguf_context_length(models[i]);

    /* Menu state */
    enum { SECTION_MODEL, SECTION_GPU, SECTION_CTX, SECTION_TEMP, SECTION_LAUNCH, SECTION_COUNT };
    int section = SECTION_MODEL;
    int model_sel = 0;
    int gpu_idx = N_GPU_OPTS - 1;  /* default: 99 */
    int ctx_val = CTX_DEFAULT;     /* free slider value */
    int temp_idx = 4;              /* default: 0.4 */

    while (1) {
        printf("\033[2J\033[H");
        printf("\033[1;36m╔══════════════════════════════════════════════════════════════╗\033[0m\n");
        printf("\033[1;36m║           BASI-CLI — Model Configuration                    ║\033[0m\n");
        printf("\033[1;36m╚══════════════════════════════════════════════════════════════╝\033[0m\n\n");

        /* Model selection */
        printf("%s MODEL %s\n",
               section == SECTION_MODEL ? "\033[1;33m▸" : "  \033[90m",
               "\033[0m");
        for (int i = 0; i < count; i++) {
            double mb = file_size_mb(models[i]);
            if (i == model_sel) {
                printf("    \033[1;36m● %s\033[90m (%.0f MB)\033[0m\n",
                       model_display_name(models[i]), mb);
            } else {
                printf("    \033[90m○ %s (%.0f MB)\033[0m\n",
                       model_display_name(models[i]), mb);
            }
        }

        printf("\n");

        /* GPU layers */
        printf("%s GPU LAYERS    \033[1m%-6d\033[0m",
               section == SECTION_GPU ? "\033[1;33m▸" : "  \033[90m",
               gpu_layer_opts[gpu_idx]);
        if (section == SECTION_GPU) printf("  \033[90m← →\033[0m");
        printf("\033[0m\n");

        /* Context size with slider bar */
        {
            int max_ctx = model_max_ctx[model_sel] > 0 ? model_max_ctx[model_sel] : 131072;
            /* Clamp ctx_val to max */
            if (ctx_val > max_ctx) ctx_val = max_ctx;

            printf("%s CONTEXT SIZE  \033[1m%-6d\033[0m",
                   section == SECTION_CTX ? "\033[1;33m▸" : "  \033[90m",
                   ctx_val);

            /* Draw slider bar */
            int bar_width = 20;
            int filled = max_ctx > 0 ? (int)((long)ctx_val * bar_width / max_ctx) : 0;
            if (filled > bar_width) filled = bar_width;
            printf(" \033[90m[");
            for (int b = 0; b < bar_width; b++) {
                if (b < filled) printf("\033[36m█");
                else printf("\033[90m░");
            }
            printf("\033[90m] max:%d\033[0m", max_ctx);
            if (section == SECTION_CTX) printf("  \033[90m← →\033[0m");
            printf("\n");
        }

        /* Temperature */
        printf("%s TEMPERATURE   \033[1m%-6.1f\033[0m",
               section == SECTION_TEMP ? "\033[1;33m▸" : "  \033[90m",
               temp_opts[temp_idx]);
        if (section == SECTION_TEMP) printf("  \033[90m← →\033[0m");
        printf("\033[0m\n");

        printf("\n");

        /* Launch button */
        if (section == SECTION_LAUNCH) {
            printf("  \033[1;32m▸ [ LAUNCH ]\033[0m\n");
        } else {
            printf("    \033[90m[ LAUNCH ]\033[0m\n");
        }

        printf("\n\033[90m↑/↓ navigate  ←/→ adjust  Enter select/launch  q quit\033[0m\n");
        fflush(stdout);

        /* Read key */
        unsigned char ch;
        if (read(STDIN_FILENO, &ch, 1) != 1) break;

        if (ch == 'q' || ch == 'Q' || ch == 3) break;

        if (ch == '\n' || ch == '\r') {
            if (section == SECTION_LAUNCH || section == SECTION_MODEL) {
                if (section == SECTION_MODEL) {
                    /* Enter on model goes to next section */
                    section = SECTION_GPU;
                    continue;
                }
                /* Launch */
                cfg.model_path = strdup(models[model_sel]);
                cfg.gpu_layers = gpu_layer_opts[gpu_idx];
                cfg.ctx_size = ctx_val;
                cfg.temperature = temp_opts[temp_idx];
                break;
            } else {
                /* Enter on setting goes to next section */
                section++;
            }
            continue;
        }

        if (ch == 27) {
            unsigned char seq[2];
            if (read(STDIN_FILENO, seq, 2) == 2 && seq[0] == '[') {
                switch (seq[1]) {
                case 'A': /* Up */
                    if (section == SECTION_MODEL && model_sel > 0) model_sel--;
                    else if (section > SECTION_MODEL) section--;
                    break;
                case 'B': /* Down */
                    if (section == SECTION_MODEL && model_sel < count - 1) model_sel++;
                    else if (section < SECTION_LAUNCH) section++;
                    break;
                case 'C': /* Right */
                    if (section == SECTION_GPU && gpu_idx < N_GPU_OPTS - 1) gpu_idx++;
                    if (section == SECTION_CTX) {
                        int max_ctx = model_max_ctx[model_sel] > 0 ? model_max_ctx[model_sel] : 131072;
                        ctx_val += CTX_STEP;
                        if (ctx_val > max_ctx) ctx_val = max_ctx;
                    }
                    if (section == SECTION_TEMP && temp_idx < N_TEMP_OPTS - 1) temp_idx++;
                    break;
                case 'D': /* Left */
                    if (section == SECTION_GPU && gpu_idx > 0) gpu_idx--;
                    if (section == SECTION_CTX) {
                        ctx_val -= CTX_STEP;
                        if (ctx_val < CTX_MIN) ctx_val = CTX_MIN;
                    }
                    if (section == SECTION_TEMP && temp_idx > 0) temp_idx--;
                    break;
                }
            }
        }
    }

    if (raw) tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig);
    printf("\033[2J\033[H");
    fflush(stdout);

    for (int i = 0; i < count; i++) free(models[i]);
    free(models);
    free(model_max_ctx);
    return cfg;
}

/* ── Main ──────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *model_path = NULL;
    int n_gpu_layers = 99;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            n_gpu_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--debug") == 0) {
            debug_mode = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("BASI-CLI - AI Chat Interface\n\n"
                   "Usage: basi-cli -m <model.gguf> [-ngl <gpu_layers>] [-d]\n\n"
                   "  -m     Path to GGUF model file\n"
                   "  -ngl   Number of GPU layers (default: 99)\n"
                   "  -d     Debug mode (show webfetch details)\n"
                   "  -h     Show this help\n\n"
                   "Environment:\n"
                   "  BASI_MODEL  Default model path if -m not specified\n\n");
            return 0;
        }
    }

    if (!model_path) {
        model_path = getenv("BASI_MODEL");
    }
    /* No model specified — show interactive picker with settings */
    static char picked_model[1024];
    int ctx_override = 0;
    float temp_override = -1;
    if (!model_path) {
        LaunchConfig cfg = pick_model();
        if (!cfg.model_path) {
            fprintf(stderr, "No model selected.\n");
            return 1;
        }
        strncpy(picked_model, cfg.model_path, sizeof(picked_model) - 1);
        free(cfg.model_path);
        model_path = picked_model;
        n_gpu_layers = cfg.gpu_layers;
        ctx_override = cfg.ctx_size;
        temp_override = cfg.temperature;
    }

    printf("BASI-CLI - Loading model...\n");
    fflush(stdout);

    llama_log_set(log_callback, NULL);
    ggml_backend_load_all();

    /* Load model */
    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    struct llama_model *model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "Error: Failed to load model from %s\n", model_path);
        return 1;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    if (!vocab) {
        fprintf(stderr, "Error: Failed to get vocabulary from model\n");
        llama_model_free(model);
        return 1;
    }

    /* Create context */
    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = ctx_override > 0 ? (uint32_t)ctx_override : CONTEXT_SIZE;
    ctx_params.n_batch = ctx_params.n_ctx;
    /* Use physical cores (half of hyperthreaded count) for CPU layers */
    int n_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (n_cores > 2) {
        int phys = n_cores / 2;  /* physical cores, no hyperthreading */
        ctx_params.n_threads = phys;
        ctx_params.n_threads_batch = phys;
    }

    struct llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Error: Failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    /* Create sampler chain */
    struct llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp_override >= 0 ? temp_override : 0.4f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    printf("Model loaded. Type your message (empty line to quit).\n\n");
    fflush(stdout);

    /* Chat messages (dynamic array) */
    struct llama_chat_message *messages = NULL;
    size_t msg_count = 0;
    size_t msg_cap   = 0;

    /* Helper to add a message */
    #define ADD_MESSAGE(role_str, content_str) do { \
        if (msg_count >= msg_cap) { \
            msg_cap = msg_cap ? msg_cap * 2 : 16; \
            messages = realloc(messages, msg_cap * sizeof(struct llama_chat_message)); \
        } \
        messages[msg_count].role = (role_str); \
        messages[msg_count].content = strdup(content_str); \
        msg_count++; \
    } while(0)

    /* System prompt with current date */
    char date_str[16];
    {
        time_t now = time(NULL);
        struct tm *t = localtime(&now);
        strftime(date_str, sizeof(date_str), "%Y-%m-%d", t);
    }
    char system_prompt[4096];
    snprintf(system_prompt, sizeof(system_prompt), SYSTEM_PROMPT_FMT, date_str);
    ADD_MESSAGE("system", system_prompt);

    char formatted_buf[FORMATTED_BUF_SZ];
    size_t prev_len = 0;

    /* REPL loop */
    while (1) {
        char *user_input = read_line("\033[32m> \033[0m");
        if (!user_input) break; /* EOF */
        if (user_input[0] == '\0') {
            free(user_input);
            continue;
        }

        ADD_MESSAGE("user", user_input);
        free(user_input);

        /* Apply chat template */
        const char *tmpl = llama_model_chat_template(model, NULL);
        int new_len = llama_chat_apply_template(
            tmpl, messages, msg_count, true,
            formatted_buf, sizeof(formatted_buf));

        if (new_len < 0) {
            printf("Error: Failed to apply chat template\n");
            fflush(stdout);
            continue;
        }

        char *prompt = formatted_buf + prev_len;
        size_t prompt_len = (size_t)new_len - prev_len;

        /* Tool execution loop */
        int tool_iterations = 0;
        const int max_tool_iterations = 5;

        while (tool_iterations < max_tool_iterations) {
            tool_iterations++;

            generation_interrupted = 0;
            setup_sigint_handler();
            GenerateResult result = generate(ctx, vocab, smpl, prompt, prompt_len);
            reset_sigint_handler();

            /* Performance metrics */
            double prompt_tps = result.prompt_time_s > 0
                ? result.prompt_tokens / result.prompt_time_s : 0;
            double gen_tps = result.gen_time_s > 0
                ? result.gen_tokens / result.gen_time_s : 0;
            printf("\033[90m[ Prompt: %.1f t/s | Generation: %.1f t/s ]\033[0m\n",
                   prompt_tps, gen_tps);
            fflush(stdout);

            /* Check for tool call */
            size_t tool_cmd_len;
            const char *tool_cmd = extract_tool_call(result.text, &tool_cmd_len);

            if (tool_cmd) {
                /* Execute tool */
                char *cmd_str = malloc(tool_cmd_len + 1);
                memcpy(cmd_str, tool_cmd, tool_cmd_len);
                cmd_str[tool_cmd_len] = '\0';

                printf("\033[90m[Executing: %s]\033[0m\n", cmd_str);
                fflush(stdout);

                char *tool_result = execute_tool(cmd_str);
                free(cmd_str);

                /* Truncate tool result if too large */
                size_t tr_len = strlen(tool_result);
                if (tr_len > MAX_TOOL_RESULT_SZ) {
                    printf("\033[90m[Truncated: %zu → %d chars]\033[0m\n",
                           tr_len, MAX_TOOL_RESULT_SZ);
                    strcpy(tool_result + MAX_TOOL_RESULT_SZ - 40,
                           "\n\n[... content truncated ...]");
                }

                /* Add assistant response */
                ADD_MESSAGE("assistant", result.text);
                free(result.text);

                /* Add tool result with context budget info */
                llama_memory_t mem = llama_get_memory(ctx);
                int used = llama_memory_seq_pos_max(mem, 0) + 1;
                int remaining = (int)llama_n_ctx(ctx) - used;

                StringBuf tool_resp;
                sb_init(&tool_resp);
                sb_append_str(&tool_resp, "<tool_result>\n");
                sb_append_str(&tool_resp, tool_result);
                char budget[128];
                snprintf(budget, sizeof(budget),
                    "\n</tool_result>\n[Context: %d/%d tokens used, %d remaining. "
                    "You MUST answer now if remaining < 8000.]",
                    used, (int)llama_n_ctx(ctx), remaining);
                sb_append_str(&tool_resp, budget);
                ADD_MESSAGE("user", sb_to_str(&tool_resp));
                sb_free(&tool_resp);
                free(tool_result);

                /* Update template for next iteration */
                int next_len = llama_chat_apply_template(
                    tmpl, messages, msg_count, true,
                    formatted_buf, sizeof(formatted_buf));
                if (next_len < 0) {
                    printf("Error: Failed to apply chat template\n");
                    fflush(stdout);
                    break;
                }
                int prev = llama_chat_apply_template(
                    tmpl, messages, msg_count - 1, false, NULL, 0);
                prompt = formatted_buf + prev;
                prompt_len = (size_t)next_len - (size_t)prev;

                printf("\n");
                fflush(stdout);
            } else {
                /* No tool call — done */
                ADD_MESSAGE("assistant", result.text);
                free(result.text);
                break;
            }
        }

        /* If we exhausted tool iterations, do one final generation for the answer */
        if (tool_iterations >= max_tool_iterations) {
            printf("\033[90m[Generating answer...]\033[0m\n");
            fflush(stdout);
            generation_interrupted = 0;
            setup_sigint_handler();
            GenerateResult final_result = generate(ctx, vocab, smpl, prompt, prompt_len);
            reset_sigint_handler();

            double gen_tps = final_result.gen_time_s > 0
                ? final_result.gen_tokens / final_result.gen_time_s : 0;
            printf("\033[90m[ Generation: %.1f t/s ]\033[0m\n", gen_tps);
            fflush(stdout);

            ADD_MESSAGE("assistant", final_result.text);
            free(final_result.text);
        }

        printf("\n");
        fflush(stdout);

        /* Update prev_len for next turn */
        int len = llama_chat_apply_template(
            tmpl, messages, msg_count, false, NULL, 0);
        if (len >= 0) prev_len = (size_t)len;
    }

    /* Cleanup */
    for (size_t i = 0; i < msg_count; i++)
        free((void *)messages[i].content);
    free(messages);
    history_free_all();

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    printf("\nGoodbye!\n");
    return 0;
}
