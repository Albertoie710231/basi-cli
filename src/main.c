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
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>

#include "llama.h"

#define MAX_TOKENS          8192
#define CONTEXT_SIZE        32768
#define MAX_FILE_TOKENS     2000
#define MAX_TOOL_RESULT_SZ  16000  /* max chars in a tool result (~4000 tokens) */
#define MAX_HISTORY         100
#define FORMATTED_BUF_SZ   (CONTEXT_SIZE * 5)

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
    "CODE TOOL (C only, requires clangd):\n"
    "- code_context <file> <symbol> : Returns ONLY clangd's structural info (signature, type, doc) for a symbol. EXACTLY two whitespace-separated arguments — no colons, no line numbers. The symbol must be a top-level identifier (function, typedef, global). Output is fenced between '=== begin clangd ===' and '=== end clangd ===' — quote that block verbatim if asked, do NOT paraphrase surrounding code as 'clangd output'. Use grep -C separately if you also want to see the code.\n"
    "  Format: <tool>code_context src/main.c execute_apply_patch</tool>\n"
    "  Use this when the user asks 'what is X', 'signature of X', or 'show me the definition of X'.\n"
    "\n"
    "SHELL TOOL:\n"
    "- bash <command> : Run an arbitrary shell command via 'bash -c'. ALWAYS requires user approval before execution; the user may deny. Use for builds (make, cargo, npm), tests, git operations, or anything not covered by the other tools. Prefer specific commands; avoid destructive operations (rm -rf, package installs) without explaining first. Output combines stdout and stderr.\n"
    "  Examples: <tool>bash make</tool>  <tool>bash git status</tool>  <tool>bash ls -la src/</tool>\n"
    "\n"
    "EDIT TOOL:\n"
    "- apply_patch : Create, modify, or delete files using a structured patch. Requires user approval. After applying, do NOT re-read the file — the result tells you success or failure. Format:\n"
    "  <tool>apply_patch\n"
    "  *** Begin Patch\n"
    "  *** Update File: <path>          (or 'Add File:' / 'Delete File:')\n"
    "  @@                                (separates hunks; only needed for >1 hunk per file)\n"
    "   context line                     (1-3 unchanged lines, prefix with single space)\n"
    "  -line to remove                   (prefix with '-')\n"
    "  +line to add                      (prefix with '+')\n"
    "  *** End Patch</tool>\n"
    "  'Add File' body is all '+'-prefixed lines (full file content). 'Delete File' has no body. Include enough surrounding ' ' context that the location is unique; if context matches multiple places the patch fails. Paths must be relative.\n"
    "  Example:\n"
    "  <tool>apply_patch\n"
    "  *** Begin Patch\n"
    "  *** Update File: src/foo.c\n"
    "  @@\n"
    "   int main(void) {\n"
    "  -    return 0;\n"
    "  +    printf(\"hello\\n\");\n"
    "  +    return 0;\n"
    "   }\n"
    "  *** End Patch</tool>\n"
    "\n"
    "SCAFFOLD TOOL:\n"
    "- scaffold <name> [<dest_dir>] : Materialize a code template into <dest_dir> (default: .). Requires user approval. Use this BEFORE apply_patch when the user asks for boilerplate (e.g. 'add a new C tool', 'create a server') — the template gives you correct structure (includes, error handling, build snippets), and you only need to apply_patch the parts that need customization. The list of available templates is in 'AVAILABLE TEMPLATES' at the end of this prompt; if you don't see one that fits, skip scaffold and write the file with apply_patch directly.\n"
    "  Examples: <tool>scaffold c-tool src/server.c</tool>  <tool>scaffold list</tool>\n"
    "\n"
    "WEB TOOL:\n"
    "- webfetch \"search query\" \"grep regex\" : Takes exactly 2 quoted arguments. First is a web search query. Second is a regex pattern (use | for OR). Fetches the top 5 results in parallel and extracts paragraphs matching the regex. Handles PDFs transparently: if a result URL ends in .pdf it is parsed as PDF, and if an HTML landing page links to a PDF, the PDF is followed and its text is inlined.\n"
    "\n"
    "LOCAL DOCUMENT TOOL:\n"
    "- readfile <path> [\"regex\"] : Read any local document (pdf, docx, odt, epub, or plain text like txt/md/source code). Optional regex narrows output to matching paragraphs; without it, dumps the first ~5000 chars. ONLY call this when the user explicitly gave you a path on their machine — NEVER invent or guess a path. If they asked about something on the web, use webfetch instead.\n"
    "\n"
    "Examples:\n"
    "<tool>head -n 50 src/main.c</tool>\n"
    "<tool>grep -n \"function main\" src/main.c</tool>\n"
    "<tool>webfetch \"latest google pixel phone specs\" \"pixel|specs|price|release|camera\"</tool>\n"
    "<tool>webfetch \"attention is all you need paper\" \"attention|transformer|self-attention|benchmark\"</tool>\n"
    "\n"
    "Tool results will appear in <tool_result> tags.\n"
    "For large files, first use 'wc' to check size, then 'head' or 'grep' to read relevant parts.\n"
    "CRITICAL RULES:\n"
    "1. After receiving tool results, you MUST answer the user immediately. Do not call another tool unless the results were completely empty.\n"
    "2. Base your answers on the actual content returned by tools. Never assume details.\n"
    "3. One webfetch call is almost always enough — PDFs linked from landing pages are auto-followed. Answer with what you have.\n"
    "4. NEVER call readfile unless the user wrote a concrete path in their own message. Do not fabricate paths like /home/user/... — that file does not exist on this machine.\n"
    "5. If a tool returns 'User denied execution.', do NOT retry. Explain in plain language why you need it and ask the user.\n"
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
#define READFILE_MAX_CHARS   5000  /* cap on extracted document text per call */
#define WEBFETCH_PDF_CHARS   2000  /* budget for an inlined PDF attached to an HTML result */

#define MINI_BROWSER "/home/alberto/Documentos/MINI_BROWSER/build/mini_browser"

static bool debug_mode = false;
static bool bash_always_allowed = false;  /* set when user picks "always" in approval prompt */
static bool apply_patch_always_allowed = false;
static bool scaffold_always_allowed = false;

typedef enum {
    PERM_DEFAULT,       /* prompt for bash, apply_patch, scaffold */
    PERM_ACCEPT_EDITS,  /* auto-approve apply_patch + scaffold; bash still prompts */
    PERM_BYPASS,        /* auto-approve everything */
} PermissionMode;
static PermissionMode permission_mode = PERM_DEFAULT;

static bool plan_mode_active = false;  /* read-only: bash/apply_patch/scaffold blocked */

static const char *perm_mode_name(PermissionMode m) {
    return m == PERM_DEFAULT      ? "default"
         : m == PERM_ACCEPT_EDITS ? "accept-edits"
         :                          "bypass";
}

/* forward decls: definitions live further down in this file. */
static int mkdir_p(const char *path);
static int request_approval(const char *tool_label, const char *cmd);

/* Awk paragraph extractor. Placeholders consumed (in order):
     %s  regex pattern
     %d  maxchars
     %d  minlen
     %d  dbg flag (0/1)
   Literal %d inside the script is written as %%d so snprintf leaves it alone. */
#define AWK_EXTRACT_PARAGRAPHS \
    "awk -v pat='%s' -v maxchars=%d -v minlen=%d -v dbg=%d '" \
    "{" \
    "  gsub(/^[[:space:]]+|[[:space:]]+$/, \"\");" \
    "  if (length($0) < minlen) {" \
    "    if (cur_len > 0) { paragraphs[++pn] = cur; cur = \"\"; cur_len = 0 }" \
    "    next" \
    "  }" \
    "  if (cur_len == 0) cur = $0; else cur = cur \"\\n\" $0;" \
    "  cur_len++;" \
    "  sz += length($0) + 1" \
    "}" \
    "END {" \
    "  if (cur_len > 0) paragraphs[++pn] = cur;" \
    "  IGNORECASE=1;" \
    "  if (sz <= 5000) {" \
    "    for (i=1; i<=pn; i++) print paragraphs[i];" \
    "    if (dbg) printf \"[DEBUG] %%d bytes, %%d paragraphs (small page, full output)\\n\", sz, pn > \"/dev/stderr\";" \
    "  } else {" \
    "    total=0; matches=0;" \
    "    for (i=1; i<=pn; i++) {" \
    "      if (total >= maxchars) break;" \
    "      if (paragraphs[i] ~ pat) {" \
    "        matches++;" \
    "        if (matches > 1) { print \"---\"; total+=4 }" \
    "        if (total >= maxchars) break;" \
    "        print paragraphs[i]; total+=length(paragraphs[i])+1;" \
    "      }" \
    "    }" \
    "    if (dbg) printf \"[DEBUG] %%d bytes page, %%d paragraphs, %%d matched, %%d bytes output\\n\", sz, pn, matches, total > \"/dev/stderr\";" \
    "  }" \
    "}'"

static bool url_has_pdf_suffix(const char *url) {
    size_t len = strlen(url);
    size_t end = len;
    for (size_t i = 0; i < len; i++) {
        if (url[i] == '?' || url[i] == '#') { end = i; break; }
    }
    if (end < 4) return false;
    return strncasecmp(url + end - 4, ".pdf", 4) == 0;
}

/* Split a URL into its scheme+host (e.g. "https://arxiv.org") and its base
   directory (URL truncated to the last '/', e.g. "https://arxiv.org/abs/").
   Relative hrefs on that page get resolved against these two prefixes. */
static void url_split_base(const char *url,
                           char *scheme_host, size_t shsz,
                           char *base_dir, size_t bdsz) {
    scheme_host[0] = '\0';
    base_dir[0]    = '\0';

    const char *sep = strstr(url, "://");
    if (!sep) return;
    const char *host_start = sep + 3;
    const char *path_start = strchr(host_start, '/');

    size_t shlen = path_start ? (size_t)(path_start - url) : strlen(url);
    if (shlen >= shsz) shlen = shsz - 1;
    memcpy(scheme_host, url, shlen);
    scheme_host[shlen] = '\0';

    if (path_start) {
        const char *last_slash = strrchr(path_start, '/');
        if (last_slash && last_slash >= path_start) {
            size_t bdlen = (size_t)(last_slash - url) + 1;
            if (bdlen >= bdsz) bdlen = bdsz - 1;
            memcpy(base_dir, url, bdlen);
            base_dir[bdlen] = '\0';
            return;
        }
    }
    snprintf(base_dir, bdsz, "%s/", scheme_host);
}

/*
 * Fetch a URL, strip HTML, then use awk to extract whole paragraphs around matches.
 *
 * A paragraph is a run of consecutive lines each of length >= minlen; any
 * shorter/blank line ends the current paragraph. The awk script:
 * - Builds the paragraph array from the stream
 * - For small pages (<=5000 usable chars), prints every paragraph
 * - For large pages, prints every paragraph whose text matches the pattern
 * - Each paragraph is emitted at most once — multiple hits inside the same
 *   paragraph do not duplicate it
 * - Separates distinct paragraphs with "---"
 * - Stops after MAX_CHARS total output (hard cap via head -c)
 */
/* fetch_and_extract: if `claim_dir` is non-NULL and non-empty, parallel
   workers coordinate PDF-fetch dedupe by atomically mkdir'ing a subdirectory
   named after the md5 of the resolved PDF URL. The first worker wins; losers
   skip the fetch. Pass NULL/empty to disable dedupe. */
static char *fetch_and_extract(const char *url, const char *pattern,
                                const char *claim_dir) {
    char cmd[16384];

    /* Debug trace lines emitted to stderr (popen only captures stdout, so
       stderr reaches the user's terminal alongside the [Fetching...] banner).
       Single quotes in URLs are already rejected at webfetch entry, so
       inlining `url` between single quotes here is safe. */
    char dbg_start[1024]  = "";
    char dbg_nopdf[256]   = "";
    if (debug_mode) {
        snprintf(dbg_start, sizeof(dbg_start),
            "echo '[wf/%s] %s' >&2; ",
            url_has_pdf_suffix(url) ? "pdf " : "html", url);
        snprintf(dbg_nopdf, sizeof(dbg_nopdf),
            "echo '[wf/html]   (no pdf link on page)' >&2; ");
    }

    /* Claim guard: either a real `if mkdir` race-gate, or a no-op `if true`.
       Hash is computed from a normalized URL so that equivalent links
       (trailing '.pdf', '/', or '?query') collapse to the same key —
       arxiv.org/pdf/ID and arxiv.org/pdf/ID.pdf resolve to the same
       document and shouldn't both be fetched. */
    char claim_begin[1024];
    char claim_end[1024];
    if (claim_dir && claim_dir[0]) {
        snprintf(claim_begin, sizeof(claim_begin),
            "pdfhash=$(printf '%%s' \"$pdfurl\" "
            " | sed -E 's|[?#].*||; s|\\.pdf$||i; s|/$||' "
            " | md5sum | cut -c1-32); "
            "if mkdir \"%s/$pdfhash\" 2>/dev/null; then %s",
            claim_dir,
            debug_mode
              ? "echo \"[wf/html]   -> pdf-follow: $pdfurl\" >&2; "
              : "");
        snprintf(claim_end, sizeof(claim_end),
            "else %s:; fi",
            debug_mode
              ? "echo \"[wf/html]   (pdf already claimed by another worker: $pdfurl)\" >&2; "
              : "");
    } else {
        snprintf(claim_begin, sizeof(claim_begin),
            "if true; then %s",
            debug_mode
              ? "echo \"[wf/html]   -> pdf-follow: $pdfurl\" >&2; "
              : "");
        snprintf(claim_end, sizeof(claim_end), "fi");
    }

    if (url_has_pdf_suffix(url)) {
        /* PDF branch: raw bytes via curl → pdftotext → paragraph extract. */
        snprintf(cmd, sizeof(cmd),
            "%s"
            "curl -sL -m 20 --max-filesize 20000000 "
            "-A 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36' "
            "'%s' 2>/dev/null "
            "| pdftotext -enc UTF-8 - - 2>/dev/null "
            "| " AWK_EXTRACT_PARAGRAPHS
            " | head -c %d",
            dbg_start,
            url, pattern, WEBFETCH_MAX_CHARS, WEBFETCH_MIN_LINE_LEN,
            debug_mode ? 1 : 0, WEBFETCH_MAX_CHARS);

        char *result = run_command(cmd, WEBFETCH_MAX_CHARS + 256);
        if (result && strlen(result) > (size_t)WEBFETCH_MAX_CHARS) {
            result[WEBFETCH_MAX_CHARS] = '\0';
        }
        return result;
    }

    /* HTML branch: mini_browser (bypasses CAPTCHAs that hit plain curl) →
       save raw HTML to a tempfile so we can both flatten it with w3m AND
       scan the hrefs for a PDF link. If one is found we follow it (one PDF
       max per landing page) and append its extracted text to the result.

       PDF-picking priority:
         1. link host matches the landing page's host (e.g. arxiv→arxiv)
         2. anchor text/href contains 'download', 'full', or 'paper'
         3. first PDF link on the page
       Relative hrefs are resolved against the landing page's scheme+host and
       base directory. */
    char scheme_host[512];
    char base_dir[1024];
    url_split_base(url, scheme_host, sizeof(scheme_host),
                       base_dir,    sizeof(base_dir));

    snprintf(cmd, sizeof(cmd),
        "%s"  /* dbg_start */
        "tmp=$(mktemp 2>/dev/null) && "
        MINI_BROWSER " --headless --timeout 15000 '%s' 2>/dev/null > \"$tmp\"; "

        /* 1) Paragraph-extract the landing page's visible text. */
        "w3m -dump -T text/html -cols 120 < \"$tmp\" 2>/dev/null "
        "| " AWK_EXTRACT_PARAGRAPHS
        " | head -c %d; "

        /* 2) Pick the best PDF link from the raw HTML. The grep catches both
              '.pdf' endings and '/pdf/' path segments (arxiv-style links
              like arxiv.org/pdf/2301.00001 have no extension). Awk then ranks
              by host match, then by 'download|full|paper' href hint, then
              first-seen. */
        "pdf=$(grep -oiE 'href=\"[^\"]*(\\.pdf|/pdf/)[^\"]*\"' \"$tmp\" 2>/dev/null "
        " | sed -E 's/^href=\"//; s/\"$//' "
        " | awk -v host='%s' '"
        "     BEGIN { IGNORECASE=1 }"
        "     {"
        "       h = $0;"
        "       if (same==\"\" && index(h, host)==1) same=h;"
        "       else if (prio==\"\" && h ~ /download|full|paper/) prio=h;"
        "       else if (first==\"\") first=h;"
        "     }"
        "     END { print (same != \"\" ? same : (prio != \"\" ? prio : first)) }'"
        "); "

        /* 3) Resolve relative URLs against scheme+host / base_dir, then fetch
              the PDF, extract, and append. Skip on empty. */
        "if [ -n \"$pdf\" ]; then "
        "  case \"$pdf\" in "
        "    http://*|https://*) pdfurl=\"$pdf\" ;; "
        "    //*)                pdfurl=\"https:$pdf\" ;; "
        "    /*)                 pdfurl='%s'\"$pdf\" ;; "
        "    *)                  pdfurl='%s'\"$pdf\" ;; "
        "  esac; "
        "  %s"  /* claim_begin — opens dedupe race-gate (or `if true`);
                   debug line for "pdf-follow" lives inside the winner side */
        "    printf '\\n--- PDF: %%s ---\\n' \"$pdfurl\"; "
        "    curl -sL -m 20 --max-filesize 20000000 "
        "      -A 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36' "
        "      \"$pdfurl\" 2>/dev/null "
        "    | pdftotext -enc UTF-8 - - 2>/dev/null "
        "    | " AWK_EXTRACT_PARAGRAPHS
        "    | head -c %d; "
        "  %s; "  /* claim_end — closes race-gate */
        "else "
        "  %s"  /* dbg_nopdf */
        "  :; "
        "fi; "
        "rm -f \"$tmp\"",
        dbg_start,
        url,
        /* 1st awk */ pattern, WEBFETCH_MAX_CHARS, WEBFETCH_MIN_LINE_LEN,
                     debug_mode ? 1 : 0, WEBFETCH_MAX_CHARS,
        /* pdf picker */ scheme_host,
        /* case resolution */ scheme_host, base_dir,
        claim_begin,
        /* 2nd awk */ pattern, WEBFETCH_PDF_CHARS, WEBFETCH_MIN_LINE_LEN,
                     debug_mode ? 1 : 0, WEBFETCH_PDF_CHARS,
        claim_end,
        dbg_nopdf);

    char *result = run_command(cmd, WEBFETCH_MAX_CHARS + WEBFETCH_PDF_CHARS + 512);
    /* Hard truncate if still too long */
    size_t hard_cap = (size_t)(WEBFETCH_MAX_CHARS + WEBFETCH_PDF_CHARS);
    if (result && strlen(result) > hard_cap) {
        result[hard_cap] = '\0';
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
        MINI_BROWSER " --headless --timeout 30000 "
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

    /* Shared claim directory so parallel workers can dedupe PDF fetches.
       If mkdtemp fails we fall back to the no-dedupe path (empty string). */
    char claim_dir[] = "/tmp/basi-wf.XXXXXX";
    if (mkdtemp(claim_dir) == NULL) {
        claim_dir[0] = '\0';
    }

    /* Fork parallel fetchers */
    int pipes[WEBFETCH_MAX_RESULTS][2];
    pid_t pids[WEBFETCH_MAX_RESULTS];

    for (int i = 0; i < url_count; i++) {
        pipe(pipes[i]);
        pids[i] = fork();
        if (pids[i] == 0) {
            close(pipes[i][0]);
            char *content = fetch_and_extract(urls[i], grep_query, claim_dir);
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
    if (claim_dir[0]) {
        char rm_cmd[300];
        snprintf(rm_cmd, sizeof(rm_cmd), "rm -rf '%s'", claim_dir);
        int rc = system(rm_cmd);
        (void)rc;
    }

    if (results.len == 0) {
        sb_free(&results);
        return strdup("No search results found.");
    }
    return sb_to_str(&results);
}

/* readfile <path> [regex]
   Reads a local document and returns its plain text. Dispatches by extension:
     .pdf        → pdftotext
     .docx       → unzip + XML strip (Word OOXML)
     .odt        → unzip + XML strip (OpenDocument)
     .epub       → unzip + XML strip across all xhtml/html files
     everything  → cat (treat as text: source code, markdown, txt, json, ...)
   Without regex, emits the first READFILE_MAX_CHARS bytes of extracted text.
   With regex, paragraph-extracts matching paragraphs (same logic as webfetch). */
static char *execute_readfile(const char *path, const char *pattern) {
    if (strncmp(path, "http://", 7) == 0 ||
        strncmp(path, "https://", 8) == 0) {
        return strdup("Error: readfile is for local paths. URLs are fetched by webfetch.");
    }
    if (strchr(path, '\'')) {
        return strdup("Error: readfile path must not contain single quotes.");
    }
    if (pattern && strchr(pattern, '\'')) {
        return strdup("Error: readfile regex must not contain single quotes.");
    }
    if (access(path, R_OK) != 0) {
        char *msg = malloc(512);
        snprintf(msg, 512, "Error: Cannot read file '%s'", path);
        return msg;
    }

    const char *ext = strrchr(path, '.');
    char extractor[2048];
    if (ext && strcasecmp(ext, ".pdf") == 0) {
        snprintf(extractor, sizeof(extractor),
            "pdftotext -enc UTF-8 '%s' - 2>/dev/null", path);
    } else if (ext && strcasecmp(ext, ".docx") == 0) {
        /* Turn paragraph closers into blank lines so the paragraph-extract
           awk still sees paragraph boundaries, then strip all remaining XML
           tags and a few common entities. */
        snprintf(extractor, sizeof(extractor),
            "unzip -p '%s' word/document.xml 2>/dev/null "
            "| sed -E 's|</w:p>|\\n\\n|g; s|<[^>]+>||g; "
            "s/&amp;/\\&/g; s/&lt;/</g; s/&gt;/>/g; s/&quot;/\"/g; s/&apos;/'\\''/g'",
            path);
    } else if (ext && strcasecmp(ext, ".odt") == 0) {
        snprintf(extractor, sizeof(extractor),
            "unzip -p '%s' content.xml 2>/dev/null "
            "| sed -E 's|</text:p>|\\n\\n|g; s|<[^>]+>||g; "
            "s/&amp;/\\&/g; s/&lt;/</g; s/&gt;/>/g; s/&quot;/\"/g; s/&apos;/'\\''/g'",
            path);
    } else if (ext && strcasecmp(ext, ".epub") == 0) {
        snprintf(extractor, sizeof(extractor),
            "unzip -p '%s' '*.xhtml' '*.html' 2>/dev/null "
            "| sed -E 's|</p>|\\n\\n|g; s|<[^>]+>||g; "
            "s/&amp;/\\&/g; s/&lt;/</g; s/&gt;/>/g; s/&quot;/\"/g; s/&apos;/'\\''/g'",
            path);
    } else {
        snprintf(extractor, sizeof(extractor), "cat '%s' 2>/dev/null", path);
    }

    char cmd[16384];
    if (pattern && pattern[0]) {
        snprintf(cmd, sizeof(cmd),
            "%s | " AWK_EXTRACT_PARAGRAPHS " | head -c %d",
            extractor, pattern, READFILE_MAX_CHARS, WEBFETCH_MIN_LINE_LEN,
            debug_mode ? 1 : 0, READFILE_MAX_CHARS);
    } else {
        snprintf(cmd, sizeof(cmd), "%s | head -c %d", extractor, READFILE_MAX_CHARS);
    }

    char *result = run_command(cmd, READFILE_MAX_CHARS + 256);
    if (result && strlen(result) > (size_t)READFILE_MAX_CHARS) {
        result[READFILE_MAX_CHARS] = '\0';
    }
    if (!result || !result[0]) {
        free(result);
        return strdup("No text extracted from file (unsupported format, scanned, or empty).");
    }
    return result;
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

static char *execute_code_context(const char *args) {
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

/* ── apply_patch: Codex-style freeform diff ───────────────────────── */

typedef enum { OP_ADD, OP_DELETE, OP_UPDATE } PatchOpKind;

typedef struct {
    char *before;   /* concatenated " "+"-" lines (with newlines); empty for OP_ADD */
    char *after;    /* concatenated " "+"+" lines */
} PatchHunk;

typedef struct {
    PatchOpKind kind;
    char       *path;
    PatchHunk  *hunks;
    int         n_hunks;
} PatchFileOp;

typedef struct {
    PatchFileOp *ops;
    int          n_ops;
    char        *error;   /* NULL on success */
} ParsedPatch;

static char *take_line(const char **p) {
    if (!**p) return NULL;
    const char *start = *p;
    const char *nl = strchr(start, '\n');
    size_t len = nl ? (size_t)(nl - start) : strlen(start);
    char *line = malloc(len + 1);
    memcpy(line, start, len);
    line[len] = '\0';
    *p = nl ? nl + 1 : start + len;
    return line;
}

static void free_parsed_patch(ParsedPatch *p) {
    if (!p) return;
    for (int i = 0; i < p->n_ops; i++) {
        free(p->ops[i].path);
        for (int j = 0; j < p->ops[i].n_hunks; j++) {
            free(p->ops[i].hunks[j].before);
            free(p->ops[i].hunks[j].after);
        }
        free(p->ops[i].hunks);
    }
    free(p->ops);
    free(p->error);
    free(p);
}

static void flush_hunk(PatchFileOp *op, StringBuf *before, StringBuf *after) {
    op->hunks = realloc(op->hunks, sizeof(PatchHunk) * (op->n_hunks + 1));
    op->hunks[op->n_hunks].before = strdup(before->len ? before->data : "");
    op->hunks[op->n_hunks].after  = strdup(after->len  ? after->data  : "");
    op->n_hunks++;
    sb_clear(before);
    sb_clear(after);
}

static ParsedPatch *parse_patch(const char *text) {
    ParsedPatch *p = calloc(1, sizeof(*p));
    int op_cap = 0;
    PatchFileOp *cur = NULL;
    StringBuf before, after;
    sb_init(&before);
    sb_init(&after);
    bool in_hunk = false;

    while (*text == ' ' || *text == '\t' || *text == '\n') text++;

    char *line = take_line(&text);
    if (!line || strcmp(line, "*** Begin Patch") != 0) {
        p->error = strdup("apply_patch: missing '*** Begin Patch' marker on first line");
        free(line);
        sb_free(&before); sb_free(&after);
        return p;
    }
    free(line);

    while ((line = take_line(&text)) != NULL) {
        if (strcmp(line, "*** End Patch") == 0) {
            free(line);
            if (in_hunk && cur) flush_hunk(cur, &before, &after);
            sb_free(&before); sb_free(&after);
            return p;
        }

        if (strncmp(line, "*** ", 4) == 0) {
            if (in_hunk && cur) { flush_hunk(cur, &before, &after); in_hunk = false; }

            if (p->n_ops >= op_cap) {
                op_cap = op_cap ? op_cap * 2 : 4;
                p->ops = realloc(p->ops, sizeof(PatchFileOp) * op_cap);
            }
            cur = &p->ops[p->n_ops++];
            memset(cur, 0, sizeof(*cur));

            if (strncmp(line, "*** Add File: ", 14) == 0) {
                cur->kind = OP_ADD;
                cur->path = strdup(line + 14);
                in_hunk = true;
            } else if (strncmp(line, "*** Delete File: ", 17) == 0) {
                cur->kind = OP_DELETE;
                cur->path = strdup(line + 17);
                in_hunk = false;
            } else if (strncmp(line, "*** Update File: ", 17) == 0) {
                cur->kind = OP_UPDATE;
                cur->path = strdup(line + 17);
                in_hunk = false;
            } else {
                p->error = malloc(256);
                snprintf(p->error, 256, "apply_patch: unknown directive: %.180s", line);
                free(line);
                sb_free(&before); sb_free(&after);
                return p;
            }
            free(line);
            continue;
        }

        if (!cur) {
            p->error = strdup("apply_patch: content before any '*** Add/Delete/Update File:' directive");
            free(line);
            sb_free(&before); sb_free(&after);
            return p;
        }

        if (cur->kind == OP_DELETE) {
            if (line[0] != '\0') {
                p->error = strdup("apply_patch: 'Delete File' takes no body lines");
                free(line);
                sb_free(&before); sb_free(&after);
                return p;
            }
            free(line);
            continue;
        }

        if (strncmp(line, "@@", 2) == 0 && cur->kind == OP_UPDATE) {
            if (in_hunk) flush_hunk(cur, &before, &after);
            in_hunk = true;
            free(line);
            continue;
        }

        char prefix = line[0];
        const char *body = (prefix == '\0') ? "" : line + 1;

        if (cur->kind == OP_ADD) {
            if (prefix != '+' && prefix != '\0') {
                p->error = malloc(256);
                snprintf(p->error, 256,
                    "apply_patch: 'Add File' line must start with '+': %.180s", line);
                free(line);
                sb_free(&before); sb_free(&after);
                return p;
            }
            sb_append_str(&after, prefix == '\0' ? "" : body);
            sb_append_char(&after, '\n');
        } else { /* OP_UPDATE */
            in_hunk = true;
            if (prefix == ' ' || prefix == '\0') {
                sb_append_str(&before, prefix == '\0' ? "" : body);
                sb_append_char(&before, '\n');
                sb_append_str(&after,  prefix == '\0' ? "" : body);
                sb_append_char(&after,  '\n');
            } else if (prefix == '-') {
                sb_append_str(&before, body);
                sb_append_char(&before, '\n');
            } else if (prefix == '+') {
                sb_append_str(&after, body);
                sb_append_char(&after, '\n');
            } else {
                p->error = malloc(256);
                snprintf(p->error, 256,
                    "apply_patch: hunk line must start with ' ', '-', or '+': %.180s", line);
                free(line);
                sb_free(&before); sb_free(&after);
                return p;
            }
        }
        free(line);
    }

    /* EOF without explicit '*** End Patch' — accept and flush. */
    if (in_hunk && cur) flush_hunk(cur, &before, &after);
    sb_free(&before); sb_free(&after);
    return p;
}

static char *apply_add_file(const char *path, const char *content) {
    struct stat st;
    if (stat(path, &st) == 0) {
        char *e = malloc(512);
        snprintf(e, 512,
            "apply_patch: '%s' already exists. Use 'Update File' instead of 'Add File'.", path);
        return e;
    }
    char *path_copy = strdup(path);
    char *dir = dirname(path_copy);
    if (strcmp(dir, ".") != 0 && strcmp(dir, "/") != 0) mkdir_p(dir);
    free(path_copy);

    FILE *f = fopen(path, "w");
    if (!f) {
        char *e = malloc(512);
        snprintf(e, 512, "apply_patch: cannot create '%s' (%s)", path, strerror(errno));
        return e;
    }
    if (content && *content) fwrite(content, 1, strlen(content), f);
    fclose(f);
    return NULL;
}

static char *apply_delete_file(const char *path) {
    if (unlink(path) != 0) {
        char *e = malloc(512);
        snprintf(e, 512, "apply_patch: cannot delete '%s' (%s)", path, strerror(errno));
        return e;
    }
    return NULL;
}

static char *apply_update_file(const char *path, PatchFileOp *op) {
    FILE *f = fopen(path, "r");
    if (!f) {
        char *e = malloc(512);
        snprintf(e, 512, "apply_patch: cannot open '%s' for update (%s)", path, strerror(errno));
        return e;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *orig = malloc(fsize + 1);
    size_t nread = fread(orig, 1, fsize, f);
    orig[nread] = '\0';
    fclose(f);

    StringBuf cur;
    sb_init(&cur);
    sb_append(&cur, orig, nread);
    free(orig);

    for (int i = 0; i < op->n_hunks; i++) {
        const char *bef = op->hunks[i].before;
        const char *aft = op->hunks[i].after;

        if (!bef || !*bef) {
            char *e = malloc(512);
            snprintf(e, 512,
                "apply_patch: hunk %d in '%s' has no context — include at least one ' ' or '-' line.",
                i + 1, path);
            sb_free(&cur);
            return e;
        }

        char *match = memmem(cur.data, cur.len, bef, strlen(bef));
        if (!match) {
            char *e = malloc(1024);
            snprintf(e, 1024,
                "apply_patch: hunk %d in '%s' — context not found. Re-read the file and verify the ' ' and '-' lines match exactly (including whitespace).",
                i + 1, path);
            sb_free(&cur);
            return e;
        }
        size_t bef_len = strlen(bef);
        char *next = memmem(match + 1, cur.len - (match - cur.data) - 1, bef, bef_len);
        if (next) {
            char *e = malloc(1024);
            snprintf(e, 1024,
                "apply_patch: hunk %d in '%s' — context matches multiple locations. Add more surrounding ' ' lines to disambiguate.",
                i + 1, path);
            sb_free(&cur);
            return e;
        }

        size_t aft_len = strlen(aft);
        size_t prefix_len = match - cur.data;
        size_t suffix_len = cur.len - prefix_len - bef_len;

        StringBuf nxt;
        sb_init(&nxt);
        sb_append(&nxt, cur.data, prefix_len);
        sb_append(&nxt, aft, aft_len);
        sb_append(&nxt, cur.data + prefix_len + bef_len, suffix_len);
        sb_free(&cur);
        cur = nxt;
    }

    f = fopen(path, "w");
    if (!f) {
        char *e = malloc(512);
        snprintf(e, 512, "apply_patch: cannot write '%s' (%s)", path, strerror(errno));
        sb_free(&cur);
        return e;
    }
    if (cur.len) fwrite(cur.data, 1, cur.len, f);
    fclose(f);
    sb_free(&cur);
    return NULL;
}

static char *execute_apply_patch(const char *patch_text) {
    ParsedPatch *p = parse_patch(patch_text);
    if (p->error) {
        char *err = strdup(p->error);
        free_parsed_patch(p);
        return err;
    }
    if (p->n_ops == 0) {
        free_parsed_patch(p);
        return strdup("apply_patch: empty patch (no Add/Delete/Update File directives)");
    }

    StringBuf summary;
    sb_init(&summary);
    for (int i = 0; i < p->n_ops; i++) {
        char line[512];
        const char *kind = p->ops[i].kind == OP_ADD    ? "add"
                         : p->ops[i].kind == OP_DELETE ? "delete"
                         :                               "edit";
        if (p->ops[i].kind == OP_UPDATE) {
            snprintf(line, sizeof(line), "%s%s %s (%d hunk%s)",
                i ? ", " : "", kind, p->ops[i].path, p->ops[i].n_hunks,
                p->ops[i].n_hunks == 1 ? "" : "s");
        } else {
            snprintf(line, sizeof(line), "%s%s %s",
                i ? ", " : "", kind, p->ops[i].path);
        }
        sb_append_str(&summary, line);
    }

    bool ap_auto = (permission_mode == PERM_BYPASS)
                || (permission_mode == PERM_ACCEPT_EDITS)
                || apply_patch_always_allowed;
    if (!ap_auto) {
        printf("\n\033[36m%s\033[0m\n", patch_text);
        char *summary_str = sb_to_str(&summary);
        int decision = request_approval("apply_patch", summary_str);
        free(summary_str);
        if (decision == 0) {
            free_parsed_patch(p);
            return strdup("User denied execution.");
        }
        if (decision == 2) apply_patch_always_allowed = true;
    } else {
        sb_free(&summary);
    }

    StringBuf result;
    sb_init(&result);
    int succeeded = 0;
    for (int i = 0; i < p->n_ops; i++) {
        char *err = NULL;
        if (p->ops[i].kind == OP_ADD) {
            const char *content = (p->ops[i].n_hunks > 0) ? p->ops[i].hunks[0].after : "";
            err = apply_add_file(p->ops[i].path, content);
        } else if (p->ops[i].kind == OP_DELETE) {
            err = apply_delete_file(p->ops[i].path);
        } else {
            err = apply_update_file(p->ops[i].path, &p->ops[i]);
        }

        if (err) {
            sb_append_str(&result, err);
            sb_append_char(&result, '\n');
            free(err);
            if (succeeded) sb_append_str(&result, "WARNING: patch was partially applied; earlier files were modified.\n");
            free_parsed_patch(p);
            return sb_to_str(&result);
        }
        char ok[512];
        const char *kind = p->ops[i].kind == OP_ADD    ? "Added"
                         : p->ops[i].kind == OP_DELETE ? "Deleted"
                         :                               "Updated";
        snprintf(ok, sizeof(ok), "%s %s\n", kind, p->ops[i].path);
        sb_append_str(&result, ok);
        succeeded++;
    }

    free_parsed_patch(p);
    return sb_to_str(&result);
}

/* ── Scaffold templates ──────────────────────────────────────────── */

typedef struct {
    char       *root;
    const char *label;
} TemplateRoot;

static int discover_template_roots(TemplateRoot roots[2]) {
    int n = 0;
    struct stat st;

    if (stat("./.basi/templates", &st) == 0 && S_ISDIR(st.st_mode)) {
        roots[n].root = strdup("./.basi/templates");
        roots[n].label = "project";
        n++;
    }
    const char *home = getenv("HOME");
    if (home) {
        char user[1024];
        snprintf(user, sizeof(user), "%s/.config/basi-cli/templates", home);
        if (stat(user, &st) == 0 && S_ISDIR(st.st_mode)) {
            roots[n].root = strdup(user);
            roots[n].label = "user";
            n++;
        }
    }
    return n;
}

static void free_template_roots(TemplateRoot roots[2], int n) {
    for (int i = 0; i < n; i++) free(roots[i].root);
}

/* Read 'description: ...' or 'post_message: ...' from <dir>/_meta. */
static char *read_meta_field(const char *template_dir, const char *field) {
    char meta_path[2048];
    snprintf(meta_path, sizeof(meta_path), "%s/_meta", template_dir);
    FILE *f = fopen(meta_path, "r");
    if (!f) return strdup("");

    size_t flen = strlen(field);
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, field, flen) == 0 && line[flen] == ':') {
            const char *v = line + flen + 1;
            while (*v == ' ' || *v == '\t') v++;
            size_t len = strlen(v);
            while (len > 0 && (v[len-1] == '\n' || v[len-1] == '\r' || v[len-1] == ' ')) len--;
            char *r = malloc(len + 1);
            memcpy(r, v, len);
            r[len] = '\0';
            fclose(f);
            return r;
        }
    }
    fclose(f);
    return strdup("");
}

/* Build a string like "  - name1 — desc1\n  - name2 — desc2\n" across all roots.
 * Returns malloc'd string. Project-scope templates shadow user-scope by name. */
static char *build_templates_index(void) {
    TemplateRoot roots[2];
    int nroots = discover_template_roots(roots);

    StringBuf sb;
    sb_init(&sb);
    char **seen = NULL;
    int n_seen = 0, cap_seen = 0;

    for (int i = 0; i < nroots; i++) {
        DIR *d = opendir(roots[i].root);
        if (!d) continue;
        struct dirent *e;
        while ((e = readdir(d))) {
            if (e->d_name[0] == '.') continue;

            bool dup = false;
            for (int j = 0; j < n_seen; j++)
                if (strcmp(seen[j], e->d_name) == 0) { dup = true; break; }
            if (dup) continue;

            char tdir[2048];
            snprintf(tdir, sizeof(tdir), "%s/%s", roots[i].root, e->d_name);
            struct stat st;
            if (stat(tdir, &st) != 0 || !S_ISDIR(st.st_mode)) continue;

            char *desc = read_meta_field(tdir, "description");
            char line[1024];
            snprintf(line, sizeof(line), "  - %s — %s\n",
                     e->d_name, *desc ? desc : "(no description)");
            sb_append_str(&sb, line);
            free(desc);

            if (n_seen >= cap_seen) {
                cap_seen = cap_seen ? cap_seen * 2 : 8;
                seen = realloc(seen, cap_seen * sizeof(char *));
            }
            seen[n_seen++] = strdup(e->d_name);
        }
        closedir(d);
    }

    for (int i = 0; i < n_seen; i++) free(seen[i]);
    free(seen);
    free_template_roots(roots, nroots);

    if (sb.len == 0) {
        sb_free(&sb);
        return strdup("  (none — to add one: mkdir -p ~/.config/basi-cli/templates/<name>, drop files inside, and add a '_meta' file with 'description: ...')\n");
    }
    return sb_to_str(&sb);
}

/* Find a template by name. Returns malloc'd absolute path or NULL. */
static char *find_template_dir(const char *name) {
    if (!name || !*name) return NULL;
    for (const char *p = name; *p; p++)
        if (*p == '/' || *p == '\\') return NULL;
    if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0) return NULL;

    TemplateRoot roots[2];
    int nroots = discover_template_roots(roots);
    char *result = NULL;
    for (int i = 0; i < nroots; i++) {
        char path[2048];
        snprintf(path, sizeof(path), "%s/%s", roots[i].root, name);
        struct stat st;
        if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {
            result = strdup(path);
            break;
        }
    }
    free_template_roots(roots, nroots);
    return result;
}

/* Recursively copy template tree (skipping '_meta'). Returns NULL on success,
 * heap error string on failure. Appends each created file path to created. */
static char *copy_template_dir(const char *src, const char *dst, StringBuf *created) {
    if (mkdir_p(dst) != 0) {
        char *e = malloc(512);
        snprintf(e, 512, "scaffold: cannot create '%s' (%s)", dst, strerror(errno));
        return e;
    }

    DIR *d = opendir(src);
    if (!d) {
        char *e = malloc(512);
        snprintf(e, 512, "scaffold: cannot open template '%s'", src);
        return e;
    }

    struct dirent *ent;
    while ((ent = readdir(d))) {
        if (ent->d_name[0] == '.') continue;
        if (strcmp(ent->d_name, "_meta") == 0) continue;

        char src_path[2048], dst_path[2048];
        snprintf(src_path, sizeof(src_path), "%s/%s", src, ent->d_name);
        snprintf(dst_path, sizeof(dst_path), "%s/%s", dst, ent->d_name);

        struct stat st;
        if (stat(src_path, &st) != 0) continue;

        if (S_ISDIR(st.st_mode)) {
            char *e = copy_template_dir(src_path, dst_path, created);
            if (e) { closedir(d); return e; }
        } else if (S_ISREG(st.st_mode)) {
            struct stat dst_st;
            if (stat(dst_path, &dst_st) == 0) {
                char *e = malloc(512);
                snprintf(e, 512, "scaffold: '%.300s' already exists; refusing to overwrite", dst_path);
                closedir(d);
                return e;
            }
            FILE *fs = fopen(src_path, "rb");
            FILE *fd = fopen(dst_path, "wb");
            if (!fs || !fd) {
                if (fs) fclose(fs);
                if (fd) fclose(fd);
                char *e = malloc(512);
                snprintf(e, 512, "scaffold: cannot copy '%.180s' -> '%.180s'", src_path, dst_path);
                closedir(d);
                return e;
            }
            char buf[4096];
            size_t n;
            while ((n = fread(buf, 1, sizeof(buf), fs)) > 0)
                fwrite(buf, 1, n, fd);
            fclose(fs);
            fclose(fd);
            sb_append_str(created, dst_path);
            sb_append_char(created, '\n');
        }
    }
    closedir(d);
    return NULL;
}

static char *execute_scaffold(const char *args) {
    while (*args == ' ' || *args == '\t' || *args == '\n') args++;
    if (!*args) {
        return strdup("Error: scaffold requires a template name. Usage: <tool>scaffold <name> [<dest>]</tool> or <tool>scaffold list</tool>");
    }

    /* scaffold list */
    if (strncmp(args, "list", 4) == 0 &&
        (args[4] == '\0' || args[4] == ' ' || args[4] == '\t' || args[4] == '\n')) {
        char *idx = build_templates_index();
        size_t need = strlen(idx) + 32;
        char *result = malloc(need);
        snprintf(result, need, "Available templates:\n%s", idx);
        free(idx);
        return result;
    }

    /* parse name */
    const char *name_end = args;
    while (*name_end && *name_end != ' ' && *name_end != '\t' && *name_end != '\n') name_end++;
    size_t name_len = name_end - args;
    char *name = malloc(name_len + 1);
    memcpy(name, args, name_len);
    name[name_len] = '\0';

    /* parse dest (default ".") */
    const char *dest_start = name_end;
    while (*dest_start == ' ' || *dest_start == '\t' || *dest_start == '\n') dest_start++;
    char *dest;
    if (*dest_start) {
        const char *dest_end = dest_start;
        while (*dest_end && *dest_end != ' ' && *dest_end != '\t' && *dest_end != '\n') dest_end++;
        size_t dest_len = dest_end - dest_start;
        dest = malloc(dest_len + 1);
        memcpy(dest, dest_start, dest_len);
        dest[dest_len] = '\0';
    } else {
        dest = strdup(".");
    }

    char *template_dir = find_template_dir(name);
    if (!template_dir) {
        char *e = malloc(512 + name_len);
        snprintf(e, 512 + name_len,
            "scaffold: template '%s' not found. Use <tool>scaffold list</tool> to see available templates.", name);
        free(name);
        free(dest);
        return e;
    }

    bool sc_auto = (permission_mode == PERM_BYPASS)
                || (permission_mode == PERM_ACCEPT_EDITS)
                || scaffold_always_allowed;
    if (!sc_auto) {
        char prompt_line[1024];
        snprintf(prompt_line, sizeof(prompt_line), "%s -> %s", name, dest);
        int decision = request_approval("scaffold", prompt_line);
        if (decision == 0) {
            free(name); free(dest); free(template_dir);
            return strdup("User denied execution.");
        }
        if (decision == 2) scaffold_always_allowed = true;
    }

    StringBuf created;
    sb_init(&created);
    char *err = copy_template_dir(template_dir, dest, &created);
    char *post = read_meta_field(template_dir, "post_message");

    StringBuf result;
    sb_init(&result);
    if (err) {
        sb_append_str(&result, err);
        free(err);
    } else {
        sb_append_str(&result, "Created files:\n");
        sb_append_str(&result, created.len ? created.data : "(none)\n");
        if (*post) {
            sb_append_str(&result, "\nNote: ");
            sb_append_str(&result, post);
            sb_append_char(&result, '\n');
        }
    }

    sb_free(&created);
    free(post);
    free(name);
    free(dest);
    free(template_dir);
    return sb_to_str(&result);
}

/* ── Approval prompt for risky tools ──────────────────────────────── */

static int request_approval(const char *tool_label, const char *cmd) {
    /* returns: 0 = deny, 1 = allow once, 2 = always allow (caller sets flag) */
    bool was_raw = raw_mode_enabled;
    if (was_raw) disable_raw_mode();

    printf("\n\033[33m>> Allow %s to run:\033[0m %s\n", tool_label, cmd);
    printf("\033[33m   [y]es / [n]o / [a]lways:\033[0m ");
    fflush(stdout);

    int decision = 0;
    char buf[16];
    if (fgets(buf, sizeof(buf), stdin)) {
        char c = buf[0];
        if (c == 'y' || c == 'Y') decision = 1;
        else if (c == 'a' || c == 'A') decision = 2;
    }

    if (was_raw) enable_raw_mode();
    return decision;
}

/* ── Execute tool command ──────────────────────────────────────────── */

static char *execute_tool(const char *command) {
    /* trim whitespace */
    while (*command == ' ' || *command == '\t' || *command == '\n') command++;
    if (!*command) return strdup("Error: Empty command");

    /* Plan mode: block bash, apply_patch, and scaffold (except 'scaffold list'). */
    if (plan_mode_active) {
        if (strncmp(command, "bash", 4) == 0 &&
            (command[4] == ' ' || command[4] == '\t' || command[4] == '\n' || command[4] == '\0')) {
            return strdup("bash not allowed in plan mode (read-only). Exit with /plan off, or write your final plan inside <plan>...</plan> tags.");
        }
        if (strncmp(command, "apply_patch", 11) == 0 &&
            (command[11] == ' ' || command[11] == '\t' || command[11] == '\n' || command[11] == '\0')) {
            return strdup("apply_patch not allowed in plan mode (read-only). Exit with /plan off, or write your final plan inside <plan>...</plan> tags.");
        }
        if (strncmp(command, "scaffold", 8) == 0 &&
            (command[8] == ' ' || command[8] == '\t' || command[8] == '\n' || command[8] == '\0')) {
            const char *a = command + 8;
            while (*a == ' ' || *a == '\t') a++;
            if (strncmp(a, "list", 4) != 0) {
                return strdup("scaffold (except 'list') not allowed in plan mode (read-only). Exit with /plan off, or write your final plan inside <plan>...</plan> tags.");
            }
        }
    }

    /* code_context: structured query via clangd LSP (read-only, no approval). */
    if (strncmp(command, "code_context", 12) == 0) {
        char after = command[12];
        if (after == ' ' || after == '\t' || after == '\n' || after == '\0') {
            return execute_code_context(command + 12);
        }
    }

    /* scaffold: copy a template tree to dest, with approval. */
    if (strncmp(command, "scaffold", 8) == 0) {
        char after = command[8];
        if (after == ' ' || after == '\t' || after == '\n' || after == '\0') {
            return execute_scaffold(command + 8);
        }
    }

    /* apply_patch: pass through entire patch body, parse + apply. */
    if (strncmp(command, "apply_patch", 11) == 0) {
        char after = command[11];
        if (after == ' ' || after == '\t' || after == '\n' || after == '\0') {
            const char *patch_text = command + 11;
            while (*patch_text == ' ' || *patch_text == '\t' || *patch_text == '\n') patch_text++;
            if (!*patch_text) {
                return strdup("Error: apply_patch requires a patch body. See system prompt for grammar.");
            }
            return execute_apply_patch(patch_text);
        }
    }

    /* bash: pass through raw command line (no tokenization), gated by approval. */
    if (strncmp(command, "bash", 4) == 0) {
        char after = command[4];
        if (after == ' ' || after == '\t' || after == '\n' || after == '\0') {
            const char *shell_cmd = command + 4;
            while (*shell_cmd == ' ' || *shell_cmd == '\t' || *shell_cmd == '\n') shell_cmd++;
            if (!*shell_cmd) {
                return strdup("Error: bash requires a command, e.g. <tool>bash ls -la</tool>");
            }

            bool auto_approve = (permission_mode == PERM_BYPASS) || bash_always_allowed;
            if (!auto_approve) {
                int decision = request_approval("bash", shell_cmd);
                if (decision == 0) return strdup("User denied execution.");
                if (decision == 2) bash_always_allowed = true;
            }

            StringBuf wrapped;
            sb_init(&wrapped);
            sb_append_str(&wrapped, "bash -c '");
            for (const char *c = shell_cmd; *c; c++) {
                if (*c == '\'') sb_append_str(&wrapped, "'\"'\"'");
                else sb_append_char(&wrapped, *c);
            }
            sb_append_str(&wrapped, "' 2>&1");
            char *result = run_command(sb_to_str(&wrapped), 512 * 1024);
            sb_free(&wrapped);
            return result;
        }
    }

    ArgList al = tokenize_command(command);
    if (al.count == 0) {
        arglist_free(&al);
        return strdup("Error: No command specified");
    }

    const char *cmd = al.args[0];

    /* Whitelist check */
    static const char *allowed[] = {
        "read", "head", "tail", "grep", "wc", "cat", "webfetch", "readfile", NULL
    };
    bool is_allowed = false;
    for (const char **a = allowed; *a; a++) {
        if (strcmp(cmd, *a) == 0) { is_allowed = true; break; }
    }
    if (!is_allowed) {
        char *msg = malloc(256);
        snprintf(msg, 256,
            "Error: Command '%s' not allowed. Use: read, head, tail, grep, wc, bash, apply_patch, scaffold, code_context, webfetch, readfile", cmd);
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

    /* Handle 'readfile' — local multi-format document reader */
    if (strcmp(cmd, "readfile") == 0) {
        if (al.count < 2) {
            arglist_free(&al);
            return strdup("Error: readfile requires a path:\n"
                          "  readfile <path> [\"regex\"]\n"
                          "Example: readfile /home/user/paper.pdf \"attention|transformer|benchmark\"");
        }
        const char *regex = (al.count >= 3) ? al.args[2] : NULL;
        char *result = execute_readfile(al.args[1], regex);
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

        /* Stop as soon as a complete <tool>...</tool> has been emitted, so the
           model can't chain dozens of speculative tool calls in one response.
           Only normal-state text is appended to `response`, so this tail match
           won't trigger inside <think> blocks. */
        if (response.len >= 7 &&
            memcmp(response.data + response.len - 7, "</tool>", 7) == 0) {
            res.gen_time_s = time_now() - timer_start;
            if (thinking_box_shown) {
                if (show_thinking) { printf("\033[0m\n"); } else { clear_thinking_box(); }
                thinking_box_shown = false;
            }
            break;
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

/*
 * Apply chat template with fallback. The C-API llama_chat_apply_template only
 * matches a hardcoded list of templates (no Jinja parser). Modern HF GGUFs ship
 * custom Jinja templates that it rejects with -1. When that happens, fall back
 * to plain ChatML, which is the de-facto format for Qwen/Phi/etc.
 */
static int apply_template(
    const char *tmpl,
    const struct llama_chat_message *msgs, size_t n_msgs,
    bool add_gen_prompt,
    char *buf, size_t buf_size)
{
    if (tmpl) {
        int r = llama_chat_apply_template(tmpl, msgs, n_msgs, add_gen_prompt, buf, buf_size);
        if (r >= 0) return r;
    }
    return apply_chatml(msgs, n_msgs, add_gen_prompt, buf, buf_size);
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

/* Recursively collect .gguf files under `root` (skips mmproj weights). */
static void scan_gguf_recursive(const char *root, char ***list, int *count, int *cap) {
    DIR *dir = opendir(root);
    if (!dir) return;
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') continue;  /* skip ., .., hidden */
        char fullpath[1024];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", root, ent->d_name);

        struct stat st;
        if (lstat(fullpath, &st) != 0) continue;

        if (S_ISDIR(st.st_mode)) {
            scan_gguf_recursive(fullpath, list, count, cap);
            continue;
        }
        /* Resolve symlinks for regular-file checks (HF stores blobs via symlinks). */
        if (S_ISLNK(st.st_mode) && stat(fullpath, &st) != 0) continue;
        if (!S_ISREG(st.st_mode) && !S_ISLNK(st.st_mode)) continue;

        size_t nlen = strlen(ent->d_name);
        if (nlen < 5) continue;
        if (strcmp(ent->d_name + nlen - 5, ".gguf") != 0) continue;
        if (strstr(ent->d_name, "mmproj") != NULL) continue;

        if (*count >= *cap) {
            *cap = *cap ? *cap * 2 : 16;
            *list = realloc(*list, *cap * sizeof(char *));
        }
        (*list)[(*count)++] = strdup(fullpath);
    }
    closedir(dir);
}

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
        snprintf(cache_dir, sizeof(cache_dir), "%s/.cache/huggingface/hub", home);
        model_search_dirs[0] = cache_dir;
    }

    /* Collect .gguf files (recursive: HF hub nests files under models--ORG--NAME/snapshots/HASH/) */
    char **models = NULL;
    int count = 0, cap = 0;

    for (int d = 0; d < MODEL_DIRS_MAX && model_search_dirs[d]; d++) {
        scan_gguf_recursive(model_search_dirs[d], &models, &count, &cap);
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

/* ── Session persistence ───────────────────────────────────────────── */

static int mkdir_p(const char *path) {
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

/* Returns malloc'd path: ~/.local/share/basi-cli/projects/<encoded-cwd>/
 * with the directory created. NULL on failure. */
static char *session_dir_path(void) {
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

static void json_escape_into(StringBuf *sb, const char *s) {
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

static void session_write_record(FILE *fp, const char *role, const char *content) {
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
static char *session_picker(const char *dir) {
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
static void session_load_into(
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

static FILE *session_open_new(const char *dir, char **out_path) {
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

    /* Running token totals for /cost */
    size_t session_prompt_tokens = 0;
    size_t session_gen_tokens    = 0;

    /* Session file (set after picker; system messages are not persisted) */
    FILE *session_fp = NULL;

    /* Helper to add a message */
    #define ADD_MESSAGE(role_str, content_str) do { \
        if (msg_count >= msg_cap) { \
            msg_cap = msg_cap ? msg_cap * 2 : 16; \
            messages = realloc(messages, msg_cap * sizeof(struct llama_chat_message)); \
        } \
        messages[msg_count].role = (role_str); \
        messages[msg_count].content = strdup(content_str); \
        if (session_fp && strcmp((role_str), "system") != 0) \
            session_write_record(session_fp, (role_str), (content_str)); \
        msg_count++; \
    } while(0)

    /* System prompt with current date */
    char date_str[16];
    {
        time_t now = time(NULL);
        struct tm *t = localtime(&now);
        strftime(date_str, sizeof(date_str), "%Y-%m-%d", t);
    }
    char system_prompt[16384];
    snprintf(system_prompt, sizeof(system_prompt), SYSTEM_PROMPT_FMT, date_str);
    {
        char *templates_idx = build_templates_index();
        size_t cur_len = strlen(system_prompt);
        snprintf(system_prompt + cur_len, sizeof(system_prompt) - cur_len,
            "\n\nAVAILABLE TEMPLATES (use the scaffold tool):\n%s", templates_idx);
        free(templates_idx);
    }
    /* Project memory: append ./BASI.md if present. Context for the model,
     * not enforcement — tools and rules above take precedence. */
    {
        FILE *bf = fopen("BASI.md", "r");
        if (bf) {
            fseek(bf, 0, SEEK_END);
            long fsize = ftell(bf);
            fseek(bf, 0, SEEK_SET);
            if (fsize > 0) {
                size_t cur_len = strlen(system_prompt);
                size_t avail = sizeof(system_prompt) - cur_len - 1;
                const char *header =
                    "\n\nPROJECT MEMORY (from ./BASI.md — facts and conventions for this codebase; tools and rules above take precedence):\n";
                size_t hlen = strlen(header);
                if (avail > hlen + 200) {
                    memcpy(system_prompt + cur_len, header, hlen);
                    cur_len += hlen;
                    avail -= hlen;
                    size_t cap = 4000;  /* ~1000 tokens; small models tune out beyond this */
                    size_t to_read = (size_t)fsize < cap ? (size_t)fsize : cap;
                    if (to_read > avail - 1) to_read = avail - 1;
                    size_t nread = fread(system_prompt + cur_len, 1, to_read, bf);
                    system_prompt[cur_len + nread] = '\0';
                    printf("\033[90m[Loaded ./BASI.md (%zu bytes%s)]\033[0m\n",
                           nread, (size_t)fsize > nread ? ", truncated" : "");
                    fflush(stdout);
                }
            }
            fclose(bf);
        }
    }
    ADD_MESSAGE("system", system_prompt);

    /* Session picker: list previous sessions for this cwd, or start a new one */
    {
        char *sess_dir = session_dir_path();
        if (sess_dir) {
            char *load_path = session_picker(sess_dir);
            if (load_path) {
                session_load_into(load_path, &messages, &msg_count, &msg_cap,
                                  (int)ctx_params.n_ctx);
                session_fp = fopen(load_path, "a");
                printf("\033[90m[Session: %s]\033[0m\n\n", load_path);
                free(load_path);
            } else {
                char *new_path = NULL;
                session_fp = session_open_new(sess_dir, &new_path);
                if (new_path) {
                    printf("\033[90m[New session: %s]\033[0m\n\n", new_path);
                    free(new_path);
                }
            }
            fflush(stdout);
            free(sess_dir);
        }
    }

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

        /* Slash commands intercept (no model call) */
        if (user_input[0] == '/') {
            if (strcmp(user_input, "/help") == 0) {
                printf(
                    "\nSlash commands:\n"
                    "  /help                 this help\n"
                    "  /clear                drop conversation history (system prompt + project memory kept)\n"
                    "  /cost                 show session token usage\n"
                    "  /save [path]          export transcript as JSONL\n"
                    "  /memory               open ./BASI.md in $EDITOR\n"
                    "  /permissions [mode]   show or set permission mode (default | accept-edits | bypass)\n"
                    "  /plan [on|off]        toggle plan mode (blocks bash/apply_patch/scaffold)\n"
                    "  /model                switch model (requires restart)\n"
                    "\n"
                    "Tools the model can call: read, head, tail, grep, wc, bash,\n"
                    "  apply_patch, scaffold, webfetch, readfile.\n\n");
                fflush(stdout);
                free(user_input);
                continue;
            }
            if (strcmp(user_input, "/clear") == 0) {
                for (size_t i = 1; i < msg_count; i++) free((void*)messages[i].content);
                msg_count = 1;
                llama_memory_clear(llama_get_memory(ctx), true);
                prev_len = 0;
                printf("\033[90m[Cleared conversation history (system prompt kept)]\033[0m\n");
                fflush(stdout);
                free(user_input);
                continue;
            }
            if (strcmp(user_input, "/cost") == 0) {
                printf("\033[90m[Session: %zu prompt tokens, %zu generated tokens, %zu total]\033[0m\n",
                       session_prompt_tokens, session_gen_tokens,
                       session_prompt_tokens + session_gen_tokens);
                fflush(stdout);
                free(user_input);
                continue;
            }
            if (strncmp(user_input, "/save", 5) == 0 &&
                (user_input[5] == '\0' || user_input[5] == ' ')) {
                const char *path = user_input + 5;
                while (*path == ' ') path++;
                char default_path[256];
                if (!*path) {
                    snprintf(default_path, sizeof(default_path),
                             "basi-transcript-%ld.jsonl", (long)time(NULL));
                    path = default_path;
                }
                FILE *out = fopen(path, "w");
                if (!out) {
                    printf("\033[31mError: cannot write '%s' (%s)\033[0m\n",
                           path, strerror(errno));
                } else {
                    for (size_t i = 0; i < msg_count; i++) {
                        fprintf(out, "{\"role\":\"%s\",\"content\":\"", messages[i].role);
                        for (const char *c = messages[i].content; *c; c++) {
                            if (*c == '"')       fputs("\\\"", out);
                            else if (*c == '\\') fputs("\\\\", out);
                            else if (*c == '\n') fputs("\\n", out);
                            else if (*c == '\r') fputs("\\r", out);
                            else if (*c == '\t') fputs("\\t", out);
                            else if ((unsigned char)*c < 32) fprintf(out, "\\u%04x", (unsigned char)*c);
                            else fputc(*c, out);
                        }
                        fputs("\"}\n", out);
                    }
                    fclose(out);
                    printf("\033[90m[Saved %zu messages to %s]\033[0m\n", msg_count, path);
                }
                fflush(stdout);
                free(user_input);
                continue;
            }
            if (strcmp(user_input, "/memory") == 0) {
                const char *editor = getenv("EDITOR");
                if (!editor || !*editor) editor = "vi";
                char cmd[256];
                snprintf(cmd, sizeof(cmd), "%s BASI.md", editor);
                bool was_raw = raw_mode_enabled;
                if (was_raw) disable_raw_mode();
                int rc = system(cmd);
                if (was_raw) enable_raw_mode();
                if (rc != 0) printf("\033[31m[Editor exited with status %d]\033[0m\n", rc);
                printf("\033[90m[BASI.md edited. Changes apply on next BASI restart.]\033[0m\n");
                fflush(stdout);
                free(user_input);
                continue;
            }
            if (strcmp(user_input, "/model") == 0) {
                printf("\033[90m[Model switching is not yet implemented — exit (Ctrl-D) and restart BASI to pick a different model.]\033[0m\n");
                fflush(stdout);
                free(user_input);
                continue;
            }
            if (strncmp(user_input, "/plan", 5) == 0 &&
                (user_input[5] == '\0' || user_input[5] == ' ')) {
                const char *arg = user_input + 5;
                while (*arg == ' ') arg++;
                if (strcmp(arg, "off") == 0) {
                    plan_mode_active = false;
                    printf("\033[90m[Plan mode: off]\033[0m\n");
                } else if (strcmp(arg, "on") == 0 || !*arg) {
                    plan_mode_active = !plan_mode_active || (*arg != '\0');
                    if (*arg) plan_mode_active = true;
                    printf(plan_mode_active
                        ? "\033[36m[Plan mode: on — bash/apply_patch/scaffold blocked. Toggle off with /plan off.]\033[0m\n"
                        : "\033[90m[Plan mode: off]\033[0m\n");
                } else {
                    printf("\033[31m[Unknown arg '%s'. Use /plan, /plan on, or /plan off]\033[0m\n", arg);
                }
                fflush(stdout);
                free(user_input);
                continue;
            }
            if (strncmp(user_input, "/permissions", 12) == 0 &&
                (user_input[12] == '\0' || user_input[12] == ' ')) {
                const char *arg = user_input + 12;
                while (*arg == ' ') arg++;
                if (!*arg) {
                    printf("\033[90m[Permission mode: %s]\033[0m\n", perm_mode_name(permission_mode));
                    printf("Modes:\n"
                           "  default       prompt before bash, apply_patch, scaffold\n"
                           "  accept-edits  auto-approve apply_patch + scaffold; bash still prompts\n"
                           "  bypass        auto-approve everything\n"
                           "Use: /permissions <mode>\n");
                } else if (strcmp(arg, "default") == 0) {
                    permission_mode = PERM_DEFAULT;
                    printf("\033[90m[Permission mode: default]\033[0m\n");
                } else if (strcmp(arg, "accept-edits") == 0) {
                    permission_mode = PERM_ACCEPT_EDITS;
                    printf("\033[90m[Permission mode: accept-edits]\033[0m\n");
                } else if (strcmp(arg, "bypass") == 0) {
                    permission_mode = PERM_BYPASS;
                    printf("\033[33m[Permission mode: bypass — all tool calls auto-approved]\033[0m\n");
                } else {
                    printf("\033[31m[Unknown mode '%s'. Try: default, accept-edits, bypass]\033[0m\n", arg);
                }
                fflush(stdout);
                free(user_input);
                continue;
            }
            printf("\033[33m[Unknown command '%s'. Type /help for the list.]\033[0m\n", user_input);
            fflush(stdout);
            free(user_input);
            continue;
        }

        if (plan_mode_active) {
            size_t blen = strlen(user_input) + 384;
            char *banner = malloc(blen);
            snprintf(banner, blen,
                "[PLAN MODE — only read tools are allowed (read, head, tail, grep, wc, webfetch, readfile). bash, apply_patch, and scaffold are BLOCKED. Reply with your final plan inside <plan>...</plan> tags; if you need clarification, ask me first instead of attempting blocked tools.]\n\n%s",
                user_input);
            ADD_MESSAGE("user", banner);
            free(banner);
        } else {
            ADD_MESSAGE("user", user_input);
        }
        /* keep user_input alive across tool iterations so the context reminder
         * can echo the original request back to the model — small models drift
         * after several tool rounds otherwise. Freed at end of turn. */

        /* Apply chat template (falls back to ChatML if model template is Jinja) */
        const char *tmpl = llama_model_chat_template(model, NULL);
        int new_len = apply_template(
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
            session_prompt_tokens += result.prompt_tokens;
            session_gen_tokens    += result.gen_tokens;

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
                char budget[512];
                snprintf(budget, sizeof(budget),
                    "\n</tool_result>\n[Context: %d/%d tokens used, %d remaining. "
                    "Original request: \"%.180s\". You MUST answer now if remaining < 8000.]",
                    used, (int)llama_n_ctx(ctx), remaining, user_input);
                sb_append_str(&tool_resp, budget);
                ADD_MESSAGE("user", sb_to_str(&tool_resp));
                sb_free(&tool_resp);
                free(tool_result);

                /* Update template for next iteration */
                int next_len = apply_template(
                    tmpl, messages, msg_count, true,
                    formatted_buf, sizeof(formatted_buf));
                if (next_len < 0) {
                    printf("Error: Failed to apply chat template\n");
                    fflush(stdout);
                    break;
                }
                int prev = apply_template(
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
            session_prompt_tokens += final_result.prompt_tokens;
            session_gen_tokens    += final_result.gen_tokens;

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
        int len = apply_template(
            tmpl, messages, msg_count, false, NULL, 0);
        if (len >= 0) prev_len = (size_t)len;

        free(user_input);
    }

    /* Cleanup */
    if (session_fp) fclose(session_fp);
    lsp_kill(&lsp);
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
