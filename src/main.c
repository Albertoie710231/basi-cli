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
#include <limits.h>
#include <sys/utsname.h>
#include <sys/ioctl.h>

#include "llama.h"

#include "util.h"
#include "globals.h"
#include "kb.h"
#include "plan.h"
#include "web.h"
#include "lsp.h"
#include "patch.h"
#include "scaffold.h"
#include "session.h"
#include "model.h"
#include "verify.h"
#include "embed.h"
#include "memory.h"
#include "deepsearch.h"
#include "chat_tmpl.h"
#include "tooldefs.h"
#include "cookbook.h"
#include "slashmenu.h"
#include "spec.h"
#include "srvgen.h"
#include "srvchat.h"

/* MAX_TOKENS, CONTEXT_SIZE → globals.h */
#define MAX_FILE_TOKENS     2000
#define MAX_TOOL_RESULT_SZ  16000  /* legacy hard ceiling (kept for reference) */
/* Tool-result truncation: keep the HEAD and the TAIL, drop the middle. Test
 * runners put first failures at the top and the pass/fail summary at the bottom,
 * so a head-only cut throws away the most useful line. Line-aware + a byte
 * ceiling (~2000 tokens) so one tool result can't dominate the context window. */
#define TOOL_RESULT_MAX_BYTES   8000
#define TOOL_RESULT_HEAD_LINES  150
#define TOOL_RESULT_TAIL_LINES  100
#define MAX_HISTORY         100
#define FORMATTED_BUF_SZ   (CONTEXT_SIZE * 5)

/* ── System prompt ─────────────────────────────────────────────────── */

static const char *SYSTEM_PROMPT_FMT =
    "You are BASI, a helpful AI assistant with access to file and web tools.\n"
    "The current date is shown at the start of each user message — trust it over any date you remember.\n"
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
    "  Format: <tool>code_context src/main.c execute_edit</tool>\n"
    "  Use this when the user asks 'what is X', 'signature of X', or 'show me the definition of X'.\n"
    "\n"
    "SHELL TOOL:\n"
    "- bash <command> : Run an arbitrary shell command via 'bash -c'. ALWAYS requires user approval before execution; the user may deny. Use for builds (make, cargo, npm), tests, git operations, or anything not covered by the other tools. Prefer specific commands; avoid destructive operations (rm -rf, package installs) without explaining first. Output combines stdout and stderr. Commands are killed after a timeout (default 120s); keep them bounded — if a test or program run is killed, your code is probably too slow or looping, so optimize it.\n"
    "  Examples: <tool>bash make</tool>  <tool>bash git status</tool>  <tool>bash ls -la src/</tool>\n"
    "\n"
    "EDIT TOOL:\n"
    "- edit <path> : Create or modify a file with one or more SEARCH/REPLACE blocks. Requires user approval. After it succeeds, do NOT re-read the file — the result tells you what happened. You MUST read the file first (read/head/grep) before editing it. Format:\n"
    "  <tool>edit <path>\n"
    "  <<<<<<< SEARCH\n"
    "  <text exactly as it currently appears in the file>\n"
    "  =======\n"
    "  <the replacement text>\n"
    "  >>>>>>> REPLACE</tool>\n"
    "  Rules: The SEARCH text is matched against the CURRENT file content — copy it verbatim (whitespace can drift a little, but the words must be exact). It must be UNIQUE; if it could match several places, include a few surrounding lines. You may stack multiple SEARCH/REPLACE blocks in one edit call (applied top-to-bottom). To CREATE a new file, leave the SEARCH section empty and put the whole file content in REPLACE. Paths are relative. To delete a file, use bash (rm).\n"
    "  Example (change a fragment — SEARCH need not be a whole line):\n"
    "  <tool>edit src/foo.c\n"
    "  <<<<<<< SEARCH\n"
    "  return 0;\n"
    "  =======\n"
    "  printf(\"hello\\n\");\n"
    "      return 0;\n"
    "  >>>>>>> REPLACE</tool>\n"
    "\n"
    "SCAFFOLD TOOL:\n"
    "- scaffold <name> [<dest_dir>] : Materialize a code template into <dest_dir> (default: .). Requires user approval. Use this BEFORE edit when the user asks for boilerplate (e.g. 'add a new C tool', 'create a server') — the template gives you correct structure (includes, error handling, build snippets), and you only need to edit the parts that need customization. The list of available templates is in 'AVAILABLE TEMPLATES' at the end of this prompt; if you don't see one that fits, skip scaffold and write the file with edit directly.\n"
    "  Examples: <tool>scaffold c-tool src/server.c</tool>  <tool>scaffold list</tool>\n"
    "\n"
    "PLANNING TOOLS (gated by plan phase):\n"
    "- assumptions (drafting only) : List unverified things the plan depends on, one '- <item>' per line. If 3 or more, code auto-routes you to a spike phase to investigate before drafting can proceed. Call this BEFORE plan_write.\n"
    "  Format: <tool>assumptions\n"
    "  - queue_free is on Node or Node2D\n"
    "  - existing damage flow uses signals or _process</tool>\n"
    "- spike_write (spike only) : Persist the spike artifact to .basi/plans/<slug>.spike.md and route the phase. Body has YAML frontmatter (slug/phase=spike/created) plus three sections: ## Question, ## Findings (each bullet citing a path or URL), ## Decision. The Decision line must be exactly one of: PROCEED-TO-PLAN | NEED-ANOTHER-SPIKE | ABANDON. Capped at 80 lines.\n"
    "- plan_write (drafting/premortem) : Save a Proposal-A3 plan artifact to .basi/plans/<slug>.md. Body must have YAML frontmatter (slug/status/created/goal) followed by seven sections in this order — ## Theme, ## Background, ## Current Condition, ## Cause Analysis, ## Target Condition, ## Implementation Plan, ## Follow-Up. Capped at 200 lines. Refuses to overwrite — pick a fresh slug to start a new plan. The frontmatter status field MUST equal the runtime phase ('drafting' or 'premortem').\n"
    "- plan_verify [<id>] (active only) : Run the verify clause of every row (or just one row by id) in the Implementation Plan table. Reports OK / FAIL / SETUP (exit 127) / SKIP (empty clause) per row, with last 5 lines of output for failures. Use after the plan has been accepted via /plan accept to confirm the implementation actually meets the verify clauses you wrote.\n"
    "  Format: <tool>plan_verify</tool>  <tool>plan_verify 1.2</tool>\n"
    "  Format: <tool>plan_write\n"
    "  ---\n"
    "  slug: <kebab-case>\n"
    "  status: drafting\n"
    "  created: YYYY-MM-DD\n"
    "  goal: <one line>\n"
    "  ---\n"
    "  # <title>\n"
    "  ## Theme\n"
    "  ...\n"
    "  ## Implementation Plan\n"
    "  | id | title | deliverable | depends_on | touches | verify | status |\n"
    "  ## Follow-Up\n"
    "  ...</tool>\n"
    "\n"
    "KNOWLEDGE BASE TOOLS (read-only, no approval; corpus lives at .basi/knowledge/):\n"
    "- docs_toc : List every document in the project's knowledge base — '<shelf>/<path>: <title>' per line, no bodies. Call this FIRST when you don't know what's in the corpus, then docs_get on a specific path. Three shelves shown in precedence order: notes (user-authored, win conflicts) > pinned (added external sources) > docs (bulk-imported official docs).\n"
    "  Format: <tool>docs_toc</tool>\n"
    "- docs_get <path>[#anchor] : Read one document or section. Path is what docs_toc showed you (e.g., docs/godot/Node.md or docs/godot/Node.md#methods/queue_free). Returns plain markdown capped at ~4 KB; if larger, the response lists available H2/H3 anchors so you can drill in.\n"
    "  Format: <tool>docs_get docs/godot/Node.md#methods/queue_free</tool>\n"
    "- docs_search <keyword> : Literal substring grep across the whole knowledge base. Returns up to 50 '<shelf>/<path>:<line>: <context>' lines. NOT semantic — matches exact substrings only. Use when you know a term but not where it lives.\n"
    "  Format: <tool>docs_search queue_free</tool>\n"
    "- docs_recent_notes : Show all user notes from this project (newest file first). Notes win over imported docs by precedence — call this BEFORE answering questions where the user might have corrected a prior assumption.\n"
    "  Format: <tool>docs_recent_notes</tool>\n"
    "- docs_vector_search <query> : Semantic similarity search across the corpus. Last-resort fallback for when docs_search (literal grep) returns nothing — embeds the query and finds chunks with the closest meaning. First call lazily loads the embedding model (one-time slow path); the embeddings sidecar caches results across calls and re-embeds only what changed. Use natural-language phrasing, not keywords. If the model isn't installed, the response tells you how to download it.\n"
    "  Format: <tool>docs_vector_search how does the planner gate tools by phase</tool>\n"
    "\n"
    "WEB TOOLS:\n"
    "- web_search \"query\" [day|week|month|year] : Search the web. ALWAYS use it for any current/latest/version/price/news/recent-events question — your training data is months stale, so NEVER answer those from memory. Returns ranked results (title + URL + snippet) AND the full extracted text of the top 3 pages. Answer from that fetched PAGE CONTENT — trust it over snippets and over your own prior knowledge (e.g. version numbers). Only call web_fetch for a different URL not already included. Optional 2nd arg filters by recency (news/latest/today).\n"
    "- web_fetch \"url\" : Fetch and extract the readable text of ONE page (http/https; a bare domain like example.com is accepted). Follows redirects safely, blocks private/internal addresses, and extracts PDFs transparently. Use after web_search, or when the user names a concrete URL.\n"
    "\n"
    "LOCAL DOCUMENT TOOL:\n"
    "- readfile <path> [\"regex\"] : Read any local document (pdf, docx, odt, epub, or plain text like txt/md/source code). Optional regex narrows output to matching paragraphs; without it, dumps the first ~5000 chars. ONLY call this when the user explicitly gave you a path on their machine — NEVER invent or guess a path. If they asked about something on the web, use web_search/web_fetch instead.\n"
    "\n"
    "Examples:\n"
    "<tool>head -n 50 src/main.c</tool>\n"
    "<tool>grep -n \"function main\" src/main.c</tool>\n"
    "<tool>web_search \"latest google pixel phone specs\"</tool>\n"
    "<tool>web_fetch \"https://arxiv.org/abs/1706.03762\"</tool>\n"
    "\n"
    "Tool results will appear in <tool_result> tags.\n"
    "For large files, first use 'wc' to check size, then 'head' or 'grep' to read relevant parts.\n"
    "CRITICAL RULES:\n"
    "1. After receiving tool results, answer the user immediately — EXCEPT you may make one follow-up call to web_fetch a promising URL returned by web_search. Otherwise, do not call another tool unless the results were completely empty.\n"
    "2. Base your answers on the actual content returned by tools. Never assume details.\n"
    "3. A web_search then a web_fetch of the best result is usually enough — don't over-fetch. Answer with what you have.\n"
    "4. NEVER call readfile unless the user wrote a concrete path in their own message. Do not fabricate paths like /home/user/... — that file does not exist on this machine.\n"
    "5. If a tool returns 'User denied execution.', do NOT retry. Explain in plain language why you need it and ask the user.\n"
    "6. Cite ONLY URLs that literally appear in the tool results. Never construct, guess, or modify a source link. If you didn't get it from a result, don't cite it.\n"
    "7. For ANY question about current or time-sensitive facts — latest/newest/current version, release, price, news, recent events, 'as of today', 'what is X now' — you MUST call web_search FIRST and answer only from the results. Your training is months out of date, so answering from memory WILL be wrong. Such a call is NECESSARY, never 'unnecessary'.\n"
    "\n"
    "Always be helpful, concise, and accurate.";

/* Slim system prompt used in NATIVE tool-calling mode: the model's own
 * template already advertises the tools (names + JSON schemas), so we drop the
 * <tool>-prose catalog and keep only behavioural steering. */
static const char *SYSTEM_PROMPT_NATIVE =
    "You are BASI, a helpful AI assistant. You have tools for reading and editing files, running shell commands, searching a local knowledge base, and searching/fetching the web — they are provided to you as callable functions.\n"
    "The current date is shown at the start of each user message — trust it over any date you remember.\n"
    "Your training data may be outdated; for anything current (latest/version/price/news/recent events) you MUST call web_search first and answer from the results.\n"
    "\n"
    "Working rules:\n"
    "- You MUST read a file (read/head/grep) before you edit it.\n"
    "- To change a file, call edit with the exact current text in 'search' and the new text in 'replace'. Do not re-read the file after a successful edit.\n"
    "- After a tool returns, base your answer on its ACTUAL output; never assume details.\n"
    "- A web_search followed by one web_fetch of the best result is usually enough — then answer.\n"
    "- Cite ONLY URLs that appear in tool results; never invent or guess links.\n"
    "- When a tool reports 'User denied execution.', do not retry; explain why you need it.\n"
    "- Reference code locations as path:line so the user can jump to them.\n"
    "Be helpful, concise, and accurate.";


/* ── Signal handling ───────────────────────────────────────────────── */

volatile sig_atomic_t generation_interrupted = 0;
volatile sig_atomic_t show_thinking = 0;
volatile sig_atomic_t generate_quiet = 0;
volatile sig_atomic_t generate_keep_think = 0;
volatile sig_atomic_t generate_native_tools = 0;
volatile sig_atomic_t generate_markdown = 0;

/* Tool-call grammar sampler (phase 2b): built once when native tools are active,
   inserted into the sampler chain, and reset before each generation (it is lazy
   and stateful — without a reset it would carry a prior turn's trigger state).
   NULL when the model's format has no tool grammar or tools are inactive. */
static struct llama_sampler *g_tool_grammar = NULL;

/* Server-backed mode: the spawned llama-server holding the weights, killed on
   exit so a 30 GB process never leaks. */
static pid_t g_srv_pid = 0;
static void kill_srv(void) { if (g_srv_pid > 0) { srvgen_kill(g_srv_pid); g_srv_pid = 0; } }

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
    /* Enable bracketed paste: the terminal now wraps pasted text in
       ESC[200~ ... ESC[201~, so the line editor can tell a typed Enter
       (submit) from a newline inside a paste (keep as content). */
    printf("\033[?2004h");
    fflush(stdout);
    return true;
}

static void disable_raw_mode(void) {
    if (raw_mode_enabled) {
        printf("\033[?2004l");   /* disable bracketed paste */
        fflush(stdout);
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
        raw_mode_enabled = false;
    }
}

/* ── Line editor ───────────────────────────────────────────────────── */

/* Recompute + repaint the slash-command dropdown for the current buffer.
 * Defined after the status-bar geometry it reads; forward-declared here. */
static void editor_menu_update(FILE *out, const char *buf, size_t len, size_t cursor,
                               int promptw, bool suppress, SlashMenuState *menu,
                               int *midx, int *mn, int *msel);

/* Replace the command token at the head of `line` with the highlighted menu
 * command (plus a trailing space when it takes arguments), close the dropdown,
 * and repaint the input line. Used by Tab and by Enter-on-a-partial-command. */
static void editor_complete(StringBuf *line, size_t *cursor, const char *prompt,
                            int promptw, SlashMenuState *menu, const int *midx,
                            int msel, bool *suppress) {
    const SlashCmd *tbl = slashmenu_table(NULL);
    const SlashCmd *c = &tbl[midx[msel]];
    size_t te = 0;
    while (te < line->len && line->data[te] != ' ') te++;   /* end of command token */

    StringBuf nb; sb_init(&nb);
    sb_append_str(&nb, c->name);
    if (c->takes_arg) sb_append_char(&nb, ' ');
    size_t newcur = nb.len;
    if (te < line->len) sb_append(&nb, line->data + te, line->len - te);
    sb_clear(line);
    sb_append(line, nb.data, nb.len);
    sb_free(&nb);

    *cursor = newcur;
    *suppress = true;
    slashmenu_close(stdout, menu, promptw, (int)*cursor);
    printf("\r\033[2K%s", prompt);                  /* redraw the input line */
    fwrite(line->data, 1, line->len, stdout);
    if (*cursor < line->len) printf("\033[%zuD", line->len - *cursor);
    fflush(stdout);
}

/* Returns malloc'd string, or NULL on EOF. Empty string on Ctrl-C. */
static char *read_line(const char *prompt) {
    /* Piped / scripted input (stdin not a TTY): the interactive char-by-char
       line editor below misreads it — ESC bytes in the data trip the arrow-key
       handler, and its raw read(fd) loop is inconsistent with the stdio (fgets)
       reads elsewhere (session picker), losing buffered lines. Use a plain
       stdio line read instead, which is consistent and control-char-safe. */
    if (!isatty(STDIN_FILENO)) {
        printf("%s", prompt);
        fflush(stdout);
        char *line = NULL;
        size_t cap = 0;
        ssize_t n = getline(&line, &cap, stdin);
        if (n < 0) { free(line); return NULL; }     /* EOF */
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
        return line;                                 /* malloc'd; caller frees */
    }

    bool raw = enable_raw_mode();

    printf("%s", prompt);
    fflush(stdout);

    StringBuf line;
    sb_init(&line);
    size_t cursor = 0;
    int hist_idx = history_count;
    bool paste_mode = false;   /* true between ESC[200~ and ESC[201~ */

    StringBuf saved;
    sb_init(&saved);

    /* slash-command dropdown state */
    SlashMenuState menu = {0};
    int  menu_idx[32], menu_n = 0, menu_sel = 0;
    bool menu_suppress = false;
    int  promptw = slashmenu_visible_width(prompt);

    while (1) {
        unsigned char ch;
        ssize_t n = slashmenu_read_byte(&ch);
        if (n <= 0) {
            /* EOF */
            if (menu.open) slashmenu_close(stdout, &menu, promptw, (int)cursor);
            if (line.len == 0) {
                sb_free(&line);
                sb_free(&saved);
                if (raw) disable_raw_mode();
                return NULL;
            }
            break;
        }

        if (ch == '\n' || ch == '\r') {
            if (paste_mode) {
                /* Newline inside a paste is content, not a submit. */
                sb_ensure(&line, 1);
                line.data[line.len++] = '\n';
                cursor = line.len;
                putchar('\n');
                continue;
            }
            /* When the dropdown is up with a partial command highlighted, Enter
               completes it instead of submitting; a fully-typed command (exact
               match) submits straight away. */
            if (menu.open && menu_n > 0) {
                const SlashCmd *tbl = slashmenu_table(NULL);
                const char *name = tbl[menu_idx[menu_sel]].name;
                size_t te = 0; while (te < line.len && line.data[te] != ' ') te++;
                bool exact = (te == strlen(name) && strncmp(line.data, name, te) == 0);
                if (!exact) {
                    editor_complete(&line, &cursor, prompt, promptw,
                                    &menu, menu_idx, menu_sel, &menu_suppress);
                    continue;
                }
            }
            if (menu.open) slashmenu_close(stdout, &menu, promptw, (int)cursor);
            printf("\n");
            fflush(stdout);
            break;
        } else if (ch == 27) {
            /* escape sequence */
            unsigned char seq[2];
            if (read(STDIN_FILENO, seq, 2) < 2) continue;
            if (seq[0] == '[') {
                if (seq[1] == '2') {
                    /* Bracketed paste markers: ESC[200~ (start) / ESC[201~
                       (end). Both begin "ESC[2"; read the 3 trailing bytes
                       to tell them apart. (Delete is ESC[3~, handled below.) */
                    unsigned char rest[3];
                    if (read(STDIN_FILENO, rest, 3) == 3 &&
                        rest[0] == '0' && rest[2] == '~') {
                        if (rest[1] == '0')      paste_mode = true;
                        else if (rest[1] == '1') { paste_mode = false; fflush(stdout); }
                    }
                    continue;
                }
                switch (seq[1]) {
                case 'A': /* Up - menu selection, else history prev */
                    if (menu.open) { if (menu_sel > 0) menu_sel--; break; }
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
                case 'B': /* Down - menu selection, else history next */
                    if (menu.open) { if (menu_sel < menu_n - 1) menu_sel++; break; }
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
                    menu_suppress = false;   /* editing re-enables the dropdown */
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
        } else if (paste_mode) {
            /* Inside a paste: take bytes literally (printable + tab),
               appended at the end. Drop other control bytes. */
            if (ch == '\t' || ch >= 32) {
                sb_ensure(&line, 1);
                line.data[line.len++] = ch;
                cursor = line.len;
                putchar(ch);
            }
        } else if (ch == 127 || ch == 8) {
            /* Backspace */
            menu_suppress = false;       /* editing re-enables the dropdown */
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
            if (menu.open) slashmenu_close(stdout, &menu, promptw, (int)cursor);
            printf("^C\n");
            fflush(stdout);
            sb_free(&line);
            sb_free(&saved);
            if (raw) disable_raw_mode();
            return strdup("");
        } else if (ch == 4) {
            /* Ctrl-D */
            if (line.len == 0) {
                if (menu.open) slashmenu_close(stdout, &menu, promptw, (int)cursor);
                sb_free(&line);
                sb_free(&saved);
                if (raw) disable_raw_mode();
                return NULL;
            }
        } else if (ch == '\t') {
            /* Tab completes the highlighted dropdown command. */
            if (menu.open)
                editor_complete(&line, &cursor, prompt, promptw,
                                &menu, menu_idx, menu_sel, &menu_suppress);
        } else if (ch >= 32) {
            /* Printable */
            menu_suppress = false;       /* editing re-enables the dropdown */
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

        /* After any edit/navigation, refresh the slash-command dropdown. Skipped
           mid-paste (a CPR round-trip would race the incoming paste bytes). */
        if (!paste_mode)
            editor_menu_update(stdout, line.data, line.len, cursor, promptw,
                               menu_suppress, &menu, menu_idx, &menu_n, &menu_sel);
    }

    if (menu.open) slashmenu_close(stdout, &menu, promptw, (int)cursor);
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


bool debug_mode = false;
bool bash_always_allowed = false;
bool apply_patch_always_allowed = false;
bool scaffold_always_allowed = false;

PermissionMode permission_mode = PERM_DEFAULT;

CompactMode compact_mode = COMPACT_RETRIEVE;  /* default; overridden by BASI_COMPACT, and
                                                 auto-falls-back to summary if no embedder */
const char *compact_mode_name(CompactMode m) {
    return m == COMPACT_OFF      ? "off"
         : m == COMPACT_RETRIEVE ? "retrieve"
         : m == COMPACT_HYBRID   ? "hybrid"
         : "summary";
}

PlanPhase plan_phase = PHASE_NONE;
char *current_plan_slug = NULL;
int spike_cycles = 0;
int spike_calls  = 0;

/* ── Read-before-edit tracker ──────────────────────────────────────── */
/* Files the model has viewed this session, canonicalised so "src/x.c",
 * "./src/x.c" and an absolute path collapse to one entry. edit reads
 * this (read_tracker_seen) to refuse editing a file that was never read. */
static char **read_paths     = NULL;
static int    read_paths_n   = 0;
static int    read_paths_cap = 0;

static char *canon_path(const char *path) {
    char buf[PATH_MAX];
    if (realpath(path, buf)) return strdup(buf);
    return strdup(path);   /* file may not exist yet; fall back to the literal */
}

void read_tracker_mark(const char *path) {
    if (!path || !*path) return;
    char *cp = canon_path(path);
    for (int i = 0; i < read_paths_n; i++)
        if (strcmp(read_paths[i], cp) == 0) { free(cp); return; }
    if (read_paths_n >= read_paths_cap) {
        read_paths_cap = read_paths_cap ? read_paths_cap * 2 : 16;
        read_paths = realloc(read_paths, sizeof(char *) * read_paths_cap);
    }
    read_paths[read_paths_n++] = cp;
}

bool read_tracker_seen(const char *path) {
    char *cp = canon_path(path);
    bool found = false;
    for (int i = 0; i < read_paths_n; i++)
        if (strcmp(read_paths[i], cp) == 0) { found = true; break; }
    free(cp);
    return found;
}

const char *perm_mode_name(PermissionMode m) {
    return m == PERM_DEFAULT      ? "default"
         : m == PERM_ACCEPT_EDITS ? "accept-edits"
         :                          "bypass";
}

const char *plan_phase_name(PlanPhase p) {
    return p == PHASE_NONE      ? "none"
         : p == PHASE_DRAFTING  ? "drafting"
         : p == PHASE_SPIKE     ? "spike"
         : p == PHASE_PREMORTEM ? "premortem"
         :                        "active";
}

const char *plan_phase_banner(PlanPhase p) {
    switch (p) {
        case PHASE_DRAFTING:
            return "[DRAFTING phase — research the task using docs_*, code_context, read/grep/wc, web_search, web_fetch, readfile. BEFORE you call plan_write, call the assumptions tool with a list (one '- <item>' per line) of every unverified thing the plan depends on; if 3 or more, you'll be auto-routed to a spike phase to investigate first. Then save the Proposal-A3 plan to .basi/plans/<slug>.md via plan_write. bash/edit/scaffold are BLOCKED (scaffold list is allowed).]";
        case PHASE_SPIKE:
            return "[SPIKE phase — read-only investigation. Allowed: docs_*, code_context, read/grep/wc, web_search, web_fetch, readfile. Budget: 12 tool calls / ~8000 tokens; stay tight. When done, call spike_write with this body:\n  ---\n  slug: <same>\n  phase: spike\n  created: YYYY-MM-DD\n  ---\n  ## Question\n  <the specific uncertainty>\n  ## Findings\n  - <bullet, each cites a path or URL>\n  ## Decision\n  PROCEED-TO-PLAN | NEED-ANOTHER-SPIKE | ABANDON\n  PROCEED-TO-PLAN returns to drafting; NEED-ANOTHER-SPIKE starts cycle N+1 (cap 3); ABANDON exits plan mode. plan_write/assumptions are blocked here.]";
        case PHASE_PREMORTEM:
            return "[PREMORTEM phase — Klein protocol. Imagine it is three months from now and this plan FAILED. Looking back, explain why. Then call plan_write to rewrite the plan with a new \"## Pre-mortem\" section containing three subsections: ### Failure modes (numbered, distinct, past-tense — \"X happened because Y\"), ### Plan revisions (bullet diff: what you changed in the plan body in response), ### Unaddressed risks (failure modes accepted as residual). The 200-line cap still holds — compress, don't truncate. web_search/web_fetch are blocked; review what's in front of you. /plan accept transitions to active.]";
        case PHASE_NONE:
        case PHASE_ACTIVE:
        default:
            return NULL;
    }
}




/* ── Approval prompt for risky tools ──────────────────────────────── */

int request_approval(const char *tool_label, const char *cmd) {
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

/* Read a small file with the `read` tool's token-budget guard. Returns
 * malloc'd content on success (and marks the path read), or a malloc'd
 * "Error: …" string. Caller frees either way. */
static char *read_small_file(const char *filepath) {
    FILE *f = fopen(filepath, "r");
    if (!f) {
        char *msg = malloc(512);
        snprintf(msg, 512, "Error: Cannot open file '%s'", filepath);
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
        return msg;
    }
    char *content = malloc(st.st_size + 1);
    if (!content) {
        fclose(f);
        return strdup("Error: out of memory reading file");
    }
    size_t nread = fread(content, 1, st.st_size, f);
    content[nread] = '\0';
    fclose(f);
    read_tracker_mark(filepath);
    return content;
}

/* Truncate an oversized tool result keeping the HEAD and the TAIL — the middle
 * of long output is usually noise, while test runners put the first failures at
 * the top and the pass/fail summary at the bottom. Line-aware (keeps at most
 * head_lines + tail_lines lines) with a hard byte ceiling. Edits in place (the
 * result is always shorter). Returns bytes dropped (0 = left unchanged). */
static size_t truncate_tool_result(char *s, int head_lines, int tail_lines,
                                   size_t max_bytes) {
    size_t len = strlen(s);
    size_t nlines = 1;
    for (size_t i = 0; i < len; i++) if (s[i] == '\n') nlines++;
    if (len <= max_bytes && nlines <= (size_t)(head_lines + tail_lines))
        return 0;                                  /* already within limits */

    /* head_end = offset just past the head_lines-th newline */
    size_t head_end = 0;
    int hl = 0;
    for (size_t i = 0; i < len && hl < head_lines; i++)
        if (s[i] == '\n') { hl++; head_end = i + 1; }
    if (head_end == 0) head_end = len;             /* no newline in head span */

    /* tail_start = offset where the last tail_lines lines begin */
    size_t tail_start = len;
    int tl = 0;
    for (size_t i = len; i-- > 0; )
        if (s[i] == '\n') { if (++tl > tail_lines) { tail_start = i + 1; break; } }

    /* enforce the byte ceiling: split the budget between head and tail */
    size_t half = max_bytes / 2;
    if (head_end > half) head_end = half;
    if (len - tail_start > half) tail_start = len - half;
    if (tail_start <= head_end) return 0;          /* nothing left to drop */

    size_t dropped_bytes = tail_start - head_end;
    int dropped_lines = 0;
    for (size_t i = head_end; i < tail_start; i++) if (s[i] == '\n') dropped_lines++;

    char marker[96];
    int mlen = snprintf(marker, sizeof marker,
        "\n[... %d lines / %zu chars truncated (head+tail kept) ...]\n",
        dropped_lines, dropped_bytes);
    if (mlen < 0) return 0;

    size_t tail_len = len - tail_start;
    size_t newlen = head_end + (size_t)mlen + tail_len;
    char *tmp = malloc(newlen + 1);
    if (!tmp) return 0;                             /* OOM: leave original intact */
    memcpy(tmp, s, head_end);
    memcpy(tmp + head_end, marker, (size_t)mlen);
    memcpy(tmp + head_end + (size_t)mlen, s + tail_start, tail_len);
    tmp[newlen] = '\0';
    memcpy(s, tmp, newlen + 1);
    free(tmp);
    return dropped_bytes;
}

/* Append one argument to a shell-command buffer, single-quote escaped so that
 * shell metacharacters — including embedded quotes and spaces — pass through
 * literally and are never re-split. Used by both the legacy tokenized path and
 * the native structured-dispatch path. */
static void sh_append_arg(StringBuf *sb, const char *arg) {
    sb_append_char(sb, ' ');
    if (strpbrk(arg, " \t\"'$`\\")) {
        sb_append_char(sb, '\'');
        for (const char *c = arg; *c; c++) {
            if (*c == '\'') sb_append_str(sb, "'\"'\"'");
            else sb_append_char(sb, *c);
        }
        sb_append_char(sb, '\'');
    } else {
        sb_append_str(sb, arg);
    }
}

static char *execute_tool(const char *command) {
    /* trim whitespace */
    while (*command == ' ' || *command == '\t' || *command == '\n') command++;
    if (!*command) return strdup("Error: Empty command");

    /* Phase gate (Decision #5): single check covers bash/edit/scaffold
     * blocking during drafting/spike/premortem AND plan_write availability. */
    if (!plan_tool_allowed(plan_phase, command)) {
        return plan_block_msg(plan_phase, command);
    }

    /* Spike-call accounting: count anything other than spike_write itself
     * while the spike is active. Done after the phase gate so we don't
     * charge for tools we already refused. */
    if (plan_phase == PHASE_SPIKE &&
        strncmp(command, "spike_write", 11) != 0) {
        spike_calls++;
    }

    /* plan_write: persist a Proposal-A3 plan artifact. */
    if (strncmp(command, "plan_write", 10) == 0) {
        char after = command[10];
        if (after == ' ' || after == '\t' || after == '\n' || after == '\0') {
            return execute_plan_write(command + 10);
        }
    }

    /* assumptions: declare unverified items; auto-routes drafting → spike. */
    if (strncmp(command, "assumptions", 11) == 0) {
        char after = command[11];
        if (after == ' ' || after == '\t' || after == '\n' || after == '\0') {
            return execute_assumptions(command + 11);
        }
    }

    /* spike_write: persist a spike artifact and route the phase. */
    if (strncmp(command, "spike_write", 11) == 0) {
        char after = command[11];
        if (after == ' ' || after == '\t' || after == '\n' || after == '\0') {
            return execute_spike_write(command + 11);
        }
    }

    /* plan_verify: run each row's verify clause, report OK/FAIL/SETUP/SKIP. */
    if (strncmp(command, "plan_verify", 11) == 0) {
        char after = command[11];
        if (after == ' ' || after == '\t' || after == '\n' || after == '\0') {
            return execute_plan_verify(command + 11);
        }
    }

    /* code_context: structured query via clangd LSP (read-only, no approval). */
    if (strncmp(command, "code_context", 12) == 0) {
        char after = command[12];
        if (after == ' ' || after == '\t' || after == '\n' || after == '\0') {
            return execute_code_context(command + 12);
        }
    }

    /* docs_* : knowledge-base librarian tools (read-only, no approval). */
    if (strncmp(command, "docs_toc", 8) == 0) {
        char after = command[8];
        if (after == ' ' || after == '\t' || after == '\n' || after == '\0') {
            return execute_docs_toc(command + 8);
        }
    }
    if (strncmp(command, "docs_get", 8) == 0) {
        char after = command[8];
        if (after == ' ' || after == '\t' || after == '\n' || after == '\0') {
            return execute_docs_get(command + 8);
        }
    }
    if (strncmp(command, "docs_search", 11) == 0) {
        char after = command[11];
        if (after == ' ' || after == '\t' || after == '\n' || after == '\0') {
            return execute_docs_search(command + 11);
        }
    }
    if (strncmp(command, "docs_recent_notes", 17) == 0) {
        char after = command[17];
        if (after == ' ' || after == '\t' || after == '\n' || after == '\0') {
            return execute_docs_recent_notes(command + 17);
        }
    }
    if (strncmp(command, "docs_vector_search", 18) == 0) {
        char after = command[18];
        if (after == ' ' || after == '\t' || after == '\n' || after == '\0') {
            return execute_docs_vector_search(command + 18);
        }
    }

    /* scaffold: copy a template tree to dest, with approval. */
    if (strncmp(command, "scaffold", 8) == 0) {
        char after = command[8];
        if (after == ' ' || after == '\t' || after == '\n' || after == '\0') {
            return execute_scaffold(command + 8);
        }
    }

    /* edit: SEARCH/REPLACE blocks; parse + apply with the fuzzy cascade. */
    if (strncmp(command, "edit", 4) == 0) {
        char after = command[4];
        if (after == ' ' || after == '\t' || after == '\n' || after == '\0') {
            const char *body = command + 4;
            while (*body == ' ' || *body == '\t' || *body == '\n') body++;
            if (!*body) {
                return strdup("Error: edit requires a file path and a SEARCH/REPLACE block. See system prompt for the format.");
            }
            return execute_edit(body);
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
            int bash_tmo = 120;            /* seconds; override via env, cap 600 */
            const char *tenv = getenv("BASI_BASH_TIMEOUT");
            if (tenv) { int v = atoi(tenv); if (v > 0) bash_tmo = v; }
            if (bash_tmo > 600) bash_tmo = 600;
            int timed_out = 0;
            char *result = run_command_timeout(sb_to_str(&wrapped), 512 * 1024,
                                               bash_tmo, &timed_out);
            sb_free(&wrapped);
            if (timed_out) {
                printf("\033[31m[bash: timed out after %ds — process group killed]\033[0m\n",
                       bash_tmo);
                fflush(stdout);
                StringBuf m;
                sb_init(&m);
                if (result) { sb_append_str(&m, result); free(result); }
                char note[320];
                snprintf(note, sizeof note,
                    "\n\n[bash: command timed out after %ds and was killed. If it is "
                    "genuinely long-running, split or simplify it. If it is a test or "
                    "program run, your code is likely too slow (e.g. exponential) or "
                    "stuck in a loop — find the inefficiency and optimize it.]", bash_tmo);
                sb_append_str(&m, note);
                return sb_to_str(&m);
            }
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
        "read", "head", "tail", "grep", "wc", "cat",
        "web_search", "web_fetch", "readfile", NULL
    };
    bool is_allowed = false;
    for (const char **a = allowed; *a; a++) {
        if (strcmp(cmd, *a) == 0) { is_allowed = true; break; }
    }
    if (!is_allowed) {
        char *msg = malloc(256);
        snprintf(msg, 256,
            "Error: Command '%s' not allowed. Use: read, head, tail, grep, wc, bash, edit, scaffold, code_context, web_search, web_fetch, readfile", cmd);
        arglist_free(&al);
        return msg;
    }

    /* Handle 'read' → file read with size check */
    if (strcmp(cmd, "read") == 0) {
        if (al.count < 2) {
            arglist_free(&al);
            return strdup("Error: read requires a file path");
        }
        char *content = read_small_file(al.args[1]);
        arglist_free(&al);
        return content;
    }

    /* Handle 'web_search' — ranked search results (no content fetch) */
    if (strcmp(cmd, "web_search") == 0) {
        if (al.count < 2) {
            arglist_free(&al);
            return strdup("Error: web_search requires a query:\n"
                          "  web_search \"search query\" [day|week|month|year]\n"
                          "Example: web_search \"google pixel 10 specs\"");
        }
        const char *time_filter = (al.count >= 3) ? al.args[2] : NULL;
        char *result = execute_web_search(al.args[1], time_filter);
        arglist_free(&al);
        return result;
    }

    /* Handle 'web_fetch' — fetch + extract one page */
    if (strcmp(cmd, "web_fetch") == 0) {
        if (al.count < 2) {
            arglist_free(&al);
            return strdup("Error: web_fetch requires a URL:\n"
                          "  web_fetch \"https://example.com/page\"\n"
                          "Example: web_fetch \"https://arxiv.org/abs/1706.03762\"");
        }
        char *result = execute_web_fetch(al.args[1]);
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
        read_tracker_mark(al.args[1]);
        arglist_free(&al);
        return result;
    }

    /* Build shell command for head/tail/grep/wc/cat */
    StringBuf shell_cmd;
    sb_init(&shell_cmd);

    const char *actual = strcmp(cmd, "read") == 0 ? "cat" : cmd;
    sb_append_str(&shell_cmd, actual);

    for (int i = 1; i < al.count; i++) {
        sh_append_arg(&shell_cmd, al.args[i]);
    }

    /* Mark any file argument as seen, so a subsequent edit to a file the
     * model inspected with head/tail/grep/wc/cat passes the read-before-edit
     * gate (these tools are how the harness reads files too large for 'read'). */
    for (int i = 1; i < al.count; i++) {
        struct stat fst;
        if (stat(al.args[i], &fst) == 0 && S_ISREG(fst.st_mode))
            read_tracker_mark(al.args[i]);
    }

    char *result = run_command(sb_to_str(&shell_cmd), 512 * 1024);
    sb_free(&shell_cmd);
    arglist_free(&al);
    return result;
}

/* ── Structured tool-turn envelopes ────────────────────────────────────
 * In native mode the assistant's tool call and the tool result are stored as
 * small JSON envelopes under the reserved roles "tool_call"/"tool_result";
 * the chat_tmpl shim decodes them into STRUCTURED common_chat messages so each
 * model's template renders the call/response pair in its own format. (Storing
 * raw text + a bare role:"tool" works for Qwen but is dropped by Gemma.) */
static char *tool_call_envelope(const char *name, const char *args_json) {
    StringBuf e; sb_init(&e);
    sb_append_str(&e, "{\"name\":");
    json_escape_into(&e, name ? name : "");
    sb_append_str(&e, ",\"arguments\":");
    sb_append_str(&e, (args_json && *args_json) ? args_json : "{}");
    sb_append_char(&e, '}');
    return sb_to_str(&e);
}
static char *tool_result_envelope(const char *name, const char *content) {
    StringBuf e; sb_init(&e);
    sb_append_str(&e, "{\"name\":");
    json_escape_into(&e, name ? name : "");
    sb_append_str(&e, ",\"content\":");
    json_escape_into(&e, content ? content : "");
    sb_append_char(&e, '}');
    return sb_to_str(&e);
}

/* ── Native function-call dispatch ─────────────────────────────────────
 * Take the model's already-parsed JSON args straight to the existing
 * handlers, bypassing basi_build_command + tokenize_command so a value that
 * contains a quote or a space (e.g. a grep pattern `foo"bar`, or a path with a
 * space) survives intact instead of being silently re-split (CODE_REVIEW_PLAN
 * Task 3). Returns a malloc'd result string, or NULL if `name` is not a known
 * tool (the caller then reports the unknown-tool error). */
static char *execute_tool_native(const char *name, const char *args_json) {
    if (!name) return NULL;
    if (!args_json) args_json = "{}";

    /* Tools dispatched directly here (parsed args → handler / escaped shell
     * command). Every other tool falls through to basi_build_command +
     * execute_tool below, which already passes free-text safely (body-style
     * tools) and re-runs the phase gate itself. The two sets are DISJOINT, so
     * the phase gate + spike accounting run exactly once per call. */
    bool direct = strcmp(name, "read") == 0  || strcmp(name, "head") == 0 ||
                  strcmp(name, "tail") == 0  || strcmp(name, "grep") == 0 ||
                  strcmp(name, "wc")   == 0  || strcmp(name, "web_search") == 0 ||
                  strcmp(name, "web_fetch") == 0 || strcmp(name, "readfile") == 0;

    if (!direct) {
        char *cmd = basi_build_command(name, args_json);
        if (!cmd) return NULL;                 /* unknown tool */
        char *r = execute_tool(cmd);            /* gate + accounting happen here */
        free(cmd);
        return r;
    }

    /* IMPORTANT: bypassing execute_tool also bypasses its phase gate + spike
     * accounting, so re-run them here keyed on the bare tool name (the gate
     * only inspects the first token). Without this, web_search/web_fetch would
     * escape the PHASE_PREMORTEM block and the spike-call budget would not
     * tick during a spike. read/head/tail/grep/wc are allowed in every phase,
     * but gating them too is harmless and keeps the invariant simple. */
    if (!plan_tool_allowed(plan_phase, name)) {
        return plan_block_msg(plan_phase, name);
    }
    if (plan_phase == PHASE_SPIKE && strcmp(name, "spike_write") != 0) {
        spike_calls++;
    }

    if (strcmp(name, "read") == 0) {
        char *file = jx_get_string(args_json, "file");
        if (!file) return strdup("Error: read requires a file path");
        char *r = read_small_file(file);
        free(file);
        return r;
    }
    if (strcmp(name, "web_search") == 0) {
        char *query = jx_get_string(args_json, "query");
        if (!query) return strdup("Error: web_search requires a query");
        char *recency = jx_get_string(args_json, "recency");   /* may be NULL */
        char *r = execute_web_search(query, recency);
        free(query); free(recency);
        return r;
    }
    if (strcmp(name, "web_fetch") == 0) {
        char *url = jx_get_string(args_json, "url");
        if (!url) return strdup("Error: web_fetch requires a url");
        char *r = execute_web_fetch(url);
        free(url);
        return r;
    }
    if (strcmp(name, "readfile") == 0) {
        char *path = jx_get_string(args_json, "path");
        if (!path) return strdup("Error: readfile requires a path");
        char *regex = jx_get_string(args_json, "regex");       /* may be NULL */
        char *r = execute_readfile(path, regex);
        read_tracker_mark(path);
        free(path); free(regex);
        return r;
    }

    /* head / tail / grep / wc — build the final shell command with proper
     * single-quote escaping (sh_append_arg), never via tokenize_command. */
    char *file = jx_get_string(args_json, "file");
    if (!file) {
        char *e = malloc(64);
        snprintf(e, 64, "Error: %s requires a file path", name);
        return e;
    }
    StringBuf sh;
    sb_init(&sh);
    if (strcmp(name, "grep") == 0) {
        sb_append_str(&sh, "grep -n");
        long ctx = jx_get_int(args_json, "context");
        if (ctx > 0) { char b[32]; snprintf(b, sizeof(b), " -C %ld", ctx); sb_append_str(&sh, b); }
        char *pattern = jx_get_string(args_json, "pattern");
        if (!pattern) { free(file); sb_free(&sh); return strdup("Error: grep requires a pattern"); }
        sh_append_arg(&sh, pattern);
        sh_append_arg(&sh, file);
        free(pattern);
    } else if (strcmp(name, "wc") == 0) {
        sb_append_str(&sh, "wc");
        sh_append_arg(&sh, file);
    } else {  /* head or tail */
        sb_append_str(&sh, name);
        long lines = jx_get_int(args_json, "lines");
        if (lines > 0) { char b[32]; snprintf(b, sizeof(b), " -n %ld", lines); sb_append_str(&sh, b); }
        sh_append_arg(&sh, file);
    }

    /* Mark the file read so a later edit passes the read-before-edit gate. */
    struct stat fst;
    if (stat(file, &fst) == 0 && S_ISREG(fst.st_mode)) read_tracker_mark(file);
    free(file);

    char *r = run_command(sb_to_str(&sh), 512 * 1024);
    sb_free(&sh);
    return r;
}



/* ── Persistent default model ──────────────────────────────────────────
 * The last model chosen via the picker or /model is remembered here, so BASI
 * drops straight into chat on every later launch instead of re-prompting.
 * Resolution order at launch: -m (explicit) > this default > $BASI_MODEL >
 * first-run picker. Best-effort: any I/O failure just falls through. */
static void default_model_dir(char *out, size_t n) {
    const char *xdg  = getenv("XDG_CONFIG_HOME");
    const char *home = getenv("HOME");
    if (xdg && *xdg)        snprintf(out, n, "%s/basi-cli", xdg);
    else if (home && *home) snprintf(out, n, "%s/.config/basi-cli", home);
    else                    snprintf(out, n, ".basi");
}

static void save_default_model(const char *path, int ngl, int ctx) {
    if (!path || !*path) return;
    char dir[512]; default_model_dir(dir, sizeof dir);
    if (mkdir(dir, 0755) != 0 && errno == ENOENT) {
        char *slash = strrchr(dir, '/');   /* create the parent (~/.config) too */
        if (slash) { *slash = '\0'; mkdir(dir, 0755); *slash = '/'; mkdir(dir, 0755); }
    }
    char file[600]; snprintf(file, sizeof file, "%s/default-model", dir);
    FILE *f = fopen(file, "w");
    if (!f) return;
    fprintf(f, "%s\n", path);
    if (ngl >= 0) fprintf(f, "ngl=%d\n", ngl);
    if (ctx >  0) fprintf(f, "ctx=%d\n", ctx);
    fclose(f);
}

/* Load the saved default into path_out (+ optional ngl/ctx). Returns true only
 * if a model was recorded AND still exists on disk (a deleted model falls
 * through to the next resolution step rather than failing the launch). */
static bool load_default_model(char *path_out, size_t n, int *ngl, int *ctx) {
    char dir[512]; default_model_dir(dir, sizeof dir);
    char file[600]; snprintf(file, sizeof file, "%s/default-model", dir);
    FILE *f = fopen(file, "r");
    if (!f) return false;
    path_out[0] = '\0';
    char line[1100];
    while (fgets(line, sizeof line, f)) {
        size_t l = strlen(line);
        while (l && (line[l-1] == '\n' || line[l-1] == '\r')) line[--l] = '\0';
        if (!line[0]) continue;
        if (strncmp(line, "ngl=", 4) == 0)      { if (ngl) *ngl = atoi(line + 4); }
        else if (strncmp(line, "ctx=", 4) == 0) { if (ctx) *ctx = atoi(line + 4); }
        else if (!path_out[0])                   snprintf(path_out, n, "%s", line);
    }
    fclose(f);
    if (!path_out[0]) return false;
    return access(path_out, R_OK) == 0;
}

/* Does a default-model config file exist at all (valid or not)? Used to decide
 * whether to auto-seed it from the first model loaded — we seed only when none
 * exists, so a one-off -m/$BASI_MODEL never clobbers a default the user set. */
static bool default_model_file_exists(void) {
    char dir[512]; default_model_dir(dir, sizeof dir);
    char file[600]; snprintf(file, sizeof file, "%s/default-model", dir);
    return access(file, F_OK) == 0;
}

/* ── CLI argument parsing ──────────────────────────────────────────── */

typedef struct {
    const char *model_path;
    int         n_gpu_layers;
    bool        ngl_set;            /* was -ngl passed explicitly? */
    const char *deepsearch_q;       /* --deepsearch: run deep research and exit */
    const char *prompt;             /* -p/--prompt: run one agent turn and exit */
    bool        no_tools;           /* --no-tools: flat completion, no tool loop */
    const char *system_override;    /* -s/--system: system prompt for --no-tools */
    int         cli_ctx;            /* -c/--ctx: context size (0 = default) */
    float       cli_temp;           /* -t/--temp: sampling temp (<0 = default) */
    int         cli_top_k;          /* --top-k: 0 = disabled */
    float       cli_top_p;          /* --top-p: 1.0 = disabled */
    uint32_t    cli_seed;           /* --seed: RNG seed for sampling */
    bool        bypass;             /* --yolo/--bypass: auto-approve all tool actions */
    const char *resume_path;        /* --resume: reload this session file, skip picker */
    bool        pick;               /* --pick: force the model picker (used by /model) */
    bool        want_exit;          /* -h/--help: caller should return exit_code */
    int         exit_code;
} Cli;

static Cli parse_args(int argc, char **argv) {
    Cli c = {
        .model_path = NULL, .n_gpu_layers = 99, .ngl_set = false,
        .deepsearch_q = NULL, .prompt = NULL, .no_tools = false,
        .system_override = NULL, .cli_ctx = 0, .cli_temp = -1.0f,
        .cli_top_k = 0, .cli_top_p = 1.0f,
        .cli_seed = LLAMA_DEFAULT_SEED, .bypass = false, .resume_path = NULL,
        .pick = false, .want_exit = false, .exit_code = 0,
    };
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            c.model_path = argv[++i];
        } else if ((strcmp(argv[i], "-ngl") == 0 || strcmp(argv[i], "--ngl") == 0)
                   && i + 1 < argc) {
            c.n_gpu_layers = atoi(argv[++i]);  /* 0 = CPU only */
            c.ngl_set = true;
        } else if ((strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--ctx") == 0)
                   && i + 1 < argc) {
            c.cli_ctx = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--temp") == 0)
                   && i + 1 < argc) {
            c.cli_temp = (float)atof(argv[++i]);
        } else if ((strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--top-k") == 0)
                   && i + 1 < argc) {
            c.cli_top_k = atoi(argv[++i]);          /* 0 = disabled */
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            c.cli_top_p = (float)atof(argv[++i]);   /* 1.0 = disabled */
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            c.cli_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--debug") == 0) {
            debug_mode = true;
        } else if ((strcmp(argv[i], "--deepsearch") == 0 || strcmp(argv[i], "-ds") == 0)
                   && i + 1 < argc) {
            c.deepsearch_q = argv[++i];
        } else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0
                    || strcmp(argv[i], "--print") == 0) && i + 1 < argc) {
            c.prompt = argv[++i];
        } else if (strcmp(argv[i], "--no-tools") == 0) {
            c.no_tools = true;
        } else if (strcmp(argv[i], "--yolo") == 0 || strcmp(argv[i], "--bypass") == 0) {
            c.bypass = true;
        } else if (strcmp(argv[i], "--resume") == 0 && i + 1 < argc) {
            c.resume_path = argv[++i];
        } else if (strcmp(argv[i], "--pick") == 0) {
            c.pick = true;
        } else if ((strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--system") == 0)
                   && i + 1 < argc) {
            c.system_override = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("BASI-CLI - AI Chat Interface\n\n"
                   "Usage:\n"
                   "  basi-cli -m <model.gguf> [-ngl <n>] [-d]          interactive session\n"
                   "  basi-cli -m <model.gguf> -p \"<prompt>\"             one-shot agent turn, prints + exits\n"
                   "  basi-cli -m <model.gguf> -p \"<prompt>\" --no-tools  one-shot flat completion (no tools)\n"
                   "  basi-cli -m <model.gguf> --deepsearch \"<question>\" one-shot deep research, prints + exits\n\n"
                   "  -m              Path to GGUF model file\n"
                   "  -ngl, --ngl     Number of model layers to offload to GPU (default: 99).\n"
                   "                  Use 0 to run entirely on CPU.\n"
                   "  -c, --ctx       Context size in tokens (default: 32768)\n"
                   "  -t, --temp      Sampling temperature (default: 0.4; 0 = greedy)\n"
                   "  -k, --top-k     Top-k sampling (default: 0 = disabled). Also BASI_TOP_K.\n"
                   "  --top-p         Top-p / nucleus sampling (default: 1.0 = disabled). Also BASI_TOP_P.\n"
                   "  --seed          RNG seed for sampling (default: random). Fix it for\n"
                   "                  reproducible output.\n"
                   "  -p, --prompt    Run a single prompt non-interactively (with tools), then exit\n"
                   "  --no-tools      With -p: one flat completion, no tool loop, no tool system prompt.\n"
                   "                  Prints only the completion to stdout (clean for scripting/data-gen).\n"
                   "  -s, --system    With --no-tools: system prompt to use (default: a minimal\n"
                   "                  \"helpful assistant\" prompt). Pass \"\" for a pure completion.\n"
                   "  --yolo, --bypass  Auto-approve ALL tool actions (bash/edit/scaffold) without\n"
                   "                  prompting. Required for -p runs that use tools (a non-interactive\n"
                   "                  approval prompt is auto-denied otherwise). Dangerous: only on\n"
                   "                  code/dirs you trust.\n"
                   "  --deepsearch    Run multi-round deep research (web + KB) non-interactively, then exit\n"
                   "  --resume <file> Reload a session file and skip the picker (used by /model)\n"
                   "  -d              Debug mode (verbose tool output)\n"
                   "  -h              Show this help\n\n"
                   "Model selection:\n"
                   "  With no -m, BASI uses the saved default (set by the first-run picker or\n"
                   "  the in-chat /model command), then $BASI_MODEL, then the picker. So after\n"
                   "  you pick once, later launches go straight to chat; /model switches later.\n\n"
                   "Environment:\n"
                   "  BASI_MODEL             Fallback model path if -m and no saved default\n"
                   "  BASI_DEEPSEARCH_ROUNDS Max deep-research rounds (default 5)\n"
                   "  BASI_DEEPSEARCH_CTX    Deep-research context size (default 32768; lower for\n"
                   "                         interactive /deepsearch on a single GPU)\n\n");
            c.want_exit = true;
            c.exit_code = 0;
            return c;
        }
    }
    return c;
}

/* ── System prompt assembly ────────────────────────────────────────── */
/* Fill `buf` (size `sz`) with the base prompt plus a static <env> identity
 * block, the scaffold template index, and (if present) a bounded slice of
 * ./BASI.md project memory. */
static void build_system_prompt(char *buf, size_t sz, bool native_tools,
                                const char *model_path) {
    snprintf(buf, sz, "%s",
             native_tools ? SYSTEM_PROMPT_NATIVE : SYSTEM_PROMPT_FMT);
    /* Environment + identity block: grounds the model in where it is running,
     * what OS, whether this is a git repo, and which model it is — removing a
     * whole class of wrong path/command/self-identity guesses. Cheap, static,
     * built once at session start. */
    {
        char cwd[PATH_MAX];
        if (!getcwd(cwd, sizeof(cwd))) snprintf(cwd, sizeof(cwd), "(unknown)");
        struct utsname uts;
        const char *osname = (uname(&uts) == 0) ? uts.sysname  : "unknown";
        const char *osrel  = (uname(&uts) == 0) ? uts.release  : "";
        const char *mname = "(local model)";
        if (model_path && *model_path) {
            const char *slash = strrchr(model_path, '/');
            mname = slash ? slash + 1 : model_path;
        }
        size_t cur_len = strlen(buf);
        snprintf(buf + cur_len, sz - cur_len,
            "\n\n<env>\n"
            "You are powered by a locally-run model loaded from: %s\n"
            "Working directory: %s\n"
            "Every tool (bash, read, edit, grep, …) ALREADY runs in this working "
            "directory. Use paths relative to it (e.g. `foo.c`, not an absolute "
            "path) and do NOT `cd` to another directory — you are already where "
            "you need to be.\n"
            "Platform: %s %s\n"
            "Is a git repo: %s\n"
            "</env>\n",
            mname, cwd, osname, osrel,
            access(".git", F_OK) == 0 ? "yes" : "no");
    }
    {
        char *templates_idx = build_templates_index();
        size_t cur_len = strlen(buf);
        snprintf(buf + cur_len, sz - cur_len,
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
                size_t cur_len = strlen(buf);
                size_t avail = sz - cur_len - 1;
                const char *header =
                    "\n\nPROJECT MEMORY (from ./BASI.md — facts and conventions for this codebase; tools and rules above take precedence):\n";
                size_t hlen = strlen(header);
                if (avail > hlen + 200) {
                    memcpy(buf + cur_len, header, hlen);
                    cur_len += hlen;
                    avail -= hlen;
                    size_t cap = 4000;  /* ~1000 tokens; small models tune out beyond this */
                    size_t to_read = (size_t)fsize < cap ? (size_t)fsize : cap;
                    if (to_read > avail - 1) to_read = avail - 1;
                    size_t nread = fread(buf + cur_len, 1, to_read, bf);
                    buf[cur_len + nread] = '\0';
                    printf("\033[90m[Loaded ./BASI.md (%zu bytes%s)]\033[0m\n",
                           nread, (size_t)fsize > nread ? ", truncated" : "");
                    fflush(stdout);
                }
            }
            fclose(bf);
        }
    }
}

/* ── Conversation message append ───────────────────────────────────── */
/* Grow the message array as needed, store the (literal) role + a strdup'd copy
 * of the content, and mirror non-system turns into the session file. Backs the
 * ADD_MESSAGE macro in main() and is called directly by the extracted REPL
 * helpers (which hold the state by pointer). */
static void repl_add_message(struct llama_chat_message **messages,
                             size_t *msg_count, size_t *msg_cap,
                             FILE *session_fp,
                             const char *role, const char *content) {
    if (*msg_count >= *msg_cap) {
        *msg_cap = *msg_cap ? *msg_cap * 2 : 16;
        *messages = realloc(*messages, *msg_cap * sizeof(struct llama_chat_message));
    }
    (*messages)[*msg_count].role    = role;
    (*messages)[*msg_count].content = strdup(content);
    if (session_fp && strcmp(role, "system") != 0)
        session_write_record(session_fp, role, content);
    (*msg_count)++;
}

/* ── Context-window occupancy tracking ─────────────────────────────────
 * The EXACT number of KV cells currently filled for the conversation
 * sequence. With llama.cpp we own the cache, so this is ground truth — not
 * the chars/4 estimate a provider-API harness is forced to use. Returns 0
 * when the cache is empty (seq_pos_max returns -1 before the first decode).
 * This measures live context OCCUPANCY (what decides the context limit),
 * which is distinct from the cumulative session token counts /cost reports
 * (those measure throughput across the whole session). */
/* Server mode holds the KV inside llama-server, not our (empty, vocab_only) local
   context — so measuring llama_memory would always report 0. Instead we track the
   exact token count of the last prompt rendered and sent to the server; the ctx
   meter, the compaction trigger, and the model's "tokens remaining" hint all read
   this. Updated after every render via srv_update_ctx_used(). */
static int basi_srv_ctx_used = 0;
static void srv_update_ctx_used(const struct llama_vocab *vocab, const char *rendered, int len) {
    if (basi_srv_port <= 0 || !vocab || !rendered || len <= 0) return;
    /* parse_special so the chat-template control tokens count as the server counts
       them; NULL/0 buffer just returns the (negated) token count needed. */
    int n = -llama_tokenize(vocab, rendered, len, NULL, 0, /*add_special*/ false, /*parse_special*/ true);
    basi_srv_ctx_used = n > 0 ? n : 0;
}

static int context_used_tokens(struct llama_context *ctx) {
    if (basi_srv_port > 0) return basi_srv_ctx_used;   /* server owns the KV, not us */
    llama_pos m = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    return m < 0 ? 0 : (int)m + 1;
}

/* Compact human token count: "830", "12.3k", "131k". */
static void fmt_token_count(char *buf, size_t n, int t) {
    if (t < 1000)        snprintf(buf, n, "%d", t);
    else if (t < 10000)  snprintf(buf, n, "%.1fk", t / 1000.0);
    else                 snprintf(buf, n, "%dk", (t + 500) / 1000);
}

/* Format a colour-graded "ctx 12.3k/32k 38%" gauge into `out`. The denominator
 * is llama_n_ctx(ctx) — the context the model was ACTUALLY loaded with, which
 * may be smaller than the requested CONTEXT_SIZE if the load OOM-retry halved
 * it. Colour is the at-a-glance warning: green < 70%, amber 70-89%, red >= 90%.
 * Trailing reset returns to the caller's dim style. */
static void format_context_meter(struct llama_context *ctx, char *out, size_t n) {
    int used  = context_used_tokens(ctx);
    int total = (int)llama_n_ctx(ctx);
    int pct   = total > 0 ? (int)((100.0 * used) / total) : 0;
    char u[16], t[16];
    fmt_token_count(u, sizeof u, used);
    fmt_token_count(t, sizeof t, total);
    const char *col = pct >= 90 ? "\033[31m" : (pct >= 70 ? "\033[33m" : "\033[32m");
    snprintf(out, n, "%sctx %s/%s %d%%\033[90m", col, u, t, pct);
}

/* ── Sticky status bar ─────────────────────────────────────────────────
 * An opencode/Hermes-style bar pinned to the terminal's bottom row, always
 * visible — including while a generation streams above it. The mechanism is a
 * DECSTBM scroll region: we reserve the last physical row by confining all
 * normal output to rows 1..R-1, so newlines scroll only the region and row R
 * stays put. The bar is repainted (save-cursor → jump to row R → write →
 * restore-cursor) on demand, so it never disturbs the cursor of whatever is
 * printing above it.
 *
 * This is display-only and strictly opt-in to a real TTY: it never enables on a
 * pipe (the benchmarks) and fully tears the region down around any screen
 * takeover ($EDITOR), on exit (atexit), and on a terminating signal — leaving a
 * stranded scroll region is the one failure that outlives the process. */
static struct {
    bool   active;
    bool   suspended;          /* torn down for a shell-out, resume afterwards */
    bool   hooks_installed;    /* atexit + signal handlers registered once */
    int    rows, cols;
    struct llama_context *ctx; /* for the live ctx meter */
    char   model_tag[32];
} g_bar = {0};

static volatile sig_atomic_t g_bar_winch = 0;
static void statusbar_on_winch(int sig) { (void)sig; g_bar_winch = 1; }

static void statusbar_query_size(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0 && ws.ws_col > 0) {
        g_bar.rows = ws.ws_row;
        g_bar.cols = ws.ws_col;
    } else {
        g_bar.rows = 24;
        g_bar.cols = 80;
    }
}

/* Build the bar text: a dark background filled to the full width, with fields
 * appended in priority order (ctx meter first, model tag last) and dropped if
 * they would overflow — so a narrow terminal degrades gracefully instead of
 * wrapping (a wrap would push a phantom line and break the pinned layout). */
static void statusbar_compose(char *out, size_t outsz) {
    int cols = g_bar.cols > 0 ? g_bar.cols : 80;
    if (cols < 8)   cols = 8;
    if (cols > 400) cols = 400;
    const int budget = cols - 1;          /* leave the last column to avoid auto-wrap */

    char inner[512];
    size_t off = 0;
    int    vis = 1;                        /* the leading bg space below */
    bool   first = true;

    #define BAR_SEP() do { \
        if (!first && vis + 3 <= budget) { \
            off += snprintf(inner + off, sizeof inner - off, "\033[38;5;240m \xc2\xb7 "); \
            vis += 3; } \
        first = false; \
    } while (0)

    /* ctx meter — always shown, colour-graded like the footer */
    {
        int used  = context_used_tokens(g_bar.ctx);
        int total = (int)llama_n_ctx(g_bar.ctx);
        int pct   = total > 0 ? (int)((100.0 * used) / total) : 0;
        char u[16], t[16];
        fmt_token_count(u, sizeof u, used);
        fmt_token_count(t, sizeof t, total);
        const char *cc = pct >= 90 ? "\033[38;5;167m"
                       : (pct >= 70 ? "\033[38;5;179m" : "\033[38;5;71m");
        char plain[48];
        int vlen = snprintf(plain, sizeof plain, "ctx %d%% %s/%s", pct, u, t);
        if (vis + vlen <= budget) {
            off += snprintf(inner + off, sizeof inner - off,
                            "\033[38;5;245mctx %s%d%%\033[38;5;245m %s/%s", cc, pct, u, t);
            vis += vlen;
            first = false;
        }
    }

    /* permission mode — only when not the safe default (a standing reminder) */
    if (permission_mode != PERM_DEFAULT) {
        const char *nm = perm_mode_name(permission_mode);
        const char *cc = permission_mode == PERM_BYPASS ? "\033[1;38;5;203m"
                                                        : "\033[38;5;179m";
        int vlen = (int)strlen(nm);
        BAR_SEP();
        if (vis + vlen <= budget) {
            off += snprintf(inner + off, sizeof inner - off, "%s%s", cc, nm);
            vis += vlen;
        }
    }

    /* memory index size — only meaningful while retrieval is the active mode */
    if (compact_mode == COMPACT_RETRIEVE || compact_mode == COMPACT_HYBRID) {
        size_t mc = mem_count();
        if (mc > 0) {
            char plain[32];
            int vlen = snprintf(plain, sizeof plain, "mem %zu", mc);
            BAR_SEP();
            if (vis + vlen <= budget) {
                off += snprintf(inner + off, sizeof inner - off, "\033[38;5;108mmem %zu", mc);
                vis += vlen;
            }
        }
    }

    /* model tag — lowest priority, first to drop on a narrow terminal */
    if (g_bar.model_tag[0]) {
        int vlen = (int)strlen(g_bar.model_tag);
        BAR_SEP();
        if (vis + vlen <= budget) {
            off += snprintf(inner + off, sizeof inner - off,
                            "\033[38;5;110m%s", g_bar.model_tag);
            vis += vlen;
        }
    }
    #undef BAR_SEP

    /* assemble: bg, a leading space, the fields, padding to full width, reset */
    int pad = budget - vis;
    if (pad < 0) pad = 0;
    size_t o = 0;
    o += snprintf(out + o, outsz - o, "\033[48;5;236m ");
    if (o + off < outsz) { memcpy(out + o, inner, off); o += off; }
    while (pad-- > 0 && o + 8 < outsz) out[o++] = ' ';
    o += snprintf(out + o, outsz - o, "\033[0m");
    out[o] = '\0';
}

static void statusbar_draw(void) {
    if (!g_bar.active) return;
    if (g_bar_winch) {                     /* terminal resized: re-reserve the row */
        g_bar_winch = 0;
        statusbar_query_size();
        printf("\033[1;%dr", g_bar.rows - 1);
    }
    char bar[1280];
    statusbar_compose(bar, sizeof bar);
    /* DECSC save → jump to status row → clear → write → DECRC restore. DECSC/DECRC
       also save/restore SGR, so a colour mid-stream above the bar is preserved. */
    printf("\0337\033[%d;1H\033[2K%s\0338", g_bar.rows, bar);
    fflush(stdout);
}

/* Reserve the bottom row and paint the bar. Shared by enable and resume. */
static void statusbar_setup_terminal(void) {
    if (!isatty(STDOUT_FILENO) || !isatty(STDIN_FILENO)) return;
    statusbar_query_size();
    if (g_bar.rows < 3) return;            /* too short to spare a row */
    /* Make a blank physical line at the bottom without clobbering content:
       go to the last row, print newline (scrolls everything up one), then set
       the scroll region and park the cursor at the bottom of it. */
    printf("\033[%d;1H\n", g_bar.rows);
    printf("\033[1;%dr", g_bar.rows - 1);
    printf("\033[%d;1H", g_bar.rows - 1);
    g_bar.active = true;
    statusbar_draw();
}

/* Async-signal-safe teardown for a terminating signal: reset the scroll region,
 * show the cursor, restore cooked mode, then re-raise to die normally. Without
 * this a kill (terminal closed, SIGTERM) would strand the scroll region. */
static void statusbar_on_fatal(int sig) {
    static const char reset[] = "\033[r\033[?25h\r\n";
    ssize_t w = write(STDOUT_FILENO, reset, sizeof reset - 1);
    (void)w;
    if (raw_mode_enabled) tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
    signal(sig, SIG_DFL);
    raise(sig);
}

static void statusbar_disable(void) {
    if (!g_bar.active) return;
    g_bar.active = false;
    printf("\033[r");                      /* reset scroll region to full screen */
    printf("\033[%d;1H\033[2K", g_bar.rows);
    fflush(stdout);
}

static void statusbar_atexit(void) { statusbar_disable(); }

static void statusbar_install_hooks(void) {
    if (g_bar.hooks_installed) return;
    g_bar.hooks_installed = true;
    atexit(statusbar_atexit);
    struct sigaction fa = {0};
    fa.sa_handler = statusbar_on_fatal;
    sigaction(SIGTERM, &fa, NULL);
    sigaction(SIGHUP,  &fa, NULL);
    sigaction(SIGQUIT, &fa, NULL);
    struct sigaction wa = {0};
    wa.sa_handler = statusbar_on_winch;
    wa.sa_flags   = SA_RESTART;
    sigaction(SIGWINCH, &wa, NULL);
}

static void statusbar_enable(struct llama_context *ctx, const char *model_tag) {
    if (!isatty(STDOUT_FILENO) || !isatty(STDIN_FILENO)) return;
    g_bar.ctx = ctx;
    snprintf(g_bar.model_tag, sizeof g_bar.model_tag, "%s", model_tag ? model_tag : "");
    statusbar_install_hooks();
    statusbar_setup_terminal();
}

/* Tear the region down for a full-screen shell-out ($EDITOR); statusbar_resume
 * re-reserves it afterward. Both no-op if the bar was never active. */
static void statusbar_suspend(void) {
    if (!g_bar.active) return;
    statusbar_disable();
    g_bar.suspended = true;
}
static void statusbar_resume(void) {
    if (!g_bar.suspended) return;
    g_bar.suspended = false;
    statusbar_setup_terminal();
}

/* Per-token hook from generate(): redraw every 8th call so the pinned ctx meter
 * climbs live during a long generation without repainting on every token. */
void statusbar_tick(void) {
    if (!g_bar.active) return;
    static unsigned n = 0;
    if ((++n & 7u) != 0u) return;
    statusbar_draw();
}

/* Recompute the slash-command dropdown for the current buffer and repaint it
 * below the input line (or close it). Placed here because it reads the status
 * bar's geometry (g_bar) to keep the menu above the reserved bottom row. */
static void editor_menu_update(FILE *out, const char *buf, size_t len, size_t cursor,
                               int promptw, bool suppress, SlashMenuState *menu,
                               int *midx, int *mn, int *msel) {
    size_t plen = 0;
    if (suppress || !slashmenu_active(buf, len, cursor, &plen)) {
        if (menu->open) slashmenu_close(out, menu, promptw, (int)cursor);
        return;
    }
    *mn = slashmenu_filter(buf + 1, plen, midx, 32);
    if (*mn == 0) {
        if (menu->open) slashmenu_close(out, menu, promptw, (int)cursor);
        return;
    }
    if (*msel >= *mn) *msel = *mn - 1;
    if (*msel < 0)    *msel = 0;

    int row = slashmenu_cpr_row();
    if (row < 1) return;                 /* terminal without CPR: skip the menu */

    struct winsize ws;
    int rows = 24, cols = 80;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0) {
        rows = ws.ws_row; cols = ws.ws_col;
    }
    int region_bottom = g_bar.active ? rows - 1 : rows;
    slashmenu_draw(out, menu, row, region_bottom, cols, midx, *mn, *msel,
                   promptw, (int)cursor);
}

/* Derive a short, lower-case model tag from the GGUF path: basename, drop the
 * .gguf suffix and a trailing quant tag (…-Q4_K_M / …-f16 / …-IQ4_XS). */
static void derive_model_tag(const char *path, char *out, size_t n) {
    if (!path || !*path) { out[0] = '\0'; return; }
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    size_t len = strlen(base);
    if (len >= n) len = n - 1;             /* bounded copy: out is small (32B) */
    memcpy(out, base, len);
    out[len] = '\0';
    char *dot = strstr(out, ".gguf");
    if (dot) *dot = '\0';
    /* strip a trailing quant token (…-Q4_K_M / -Q6_K / -IQ4_XS / -f16 / -bf16),
       requiring a digit so a real suffix like "-base"/"-flash" is left alone. */
    char *dash = strrchr(out, '-');
    if (dash) {
        const char *s = dash + 1;
        bool is_quant =
            (s[0] == 'Q' && s[1] >= '0' && s[1] <= '9') ||
            (s[0] == 'I' && s[1] == 'Q' && s[2] >= '0' && s[2] <= '9') ||
            strcmp(s, "f16") == 0 || strcmp(s, "bf16") == 0 || strcmp(s, "f32") == 0;
        if (is_quant) *dash = '\0';
    }
    for (char *p = out; *p; p++)
        if (*p >= 'A' && *p <= 'Z') *p += 32;
}

/* ── Tool-activity display ─────────────────────────────────────────────
 * Render each tool call as a colour-graded bullet + tool name + its key
 * argument (the file/command/query — which the old flat "[Executing: read]"
 * never showed), with the outcome folded in as a dim indented sub-line.
 * Everything is display-only chrome; it never touches what the model sees. */

/* Display columns of a UTF-8 string: count bytes that aren't continuation
 * bytes (0x80–0xBF). Correct for the box-drawing glyphs and "·" used here,
 * which are multi-byte but one column wide. */
static size_t disp_width(const char *s) {
    size_t w = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++)
        if ((*p & 0xC0) != 0x80) w++;
    return w;
}

/* Collapse whitespace runs to single spaces (trimming ends) and ellipsize to
 * `maxw` display columns, in place and UTF-8-safe. Turns a multi-line shell
 * command or a long path into one tidy argument fragment. */
static void tidy_arg(char *s, int maxw) {
    if (!s) return;
    char *w = s;
    bool prev_sp = false, started = false;
    for (char *r = s; *r; r++) {
        unsigned char c = (unsigned char)*r;
        if (c == '\n' || c == '\t' || c == '\r' || c == ' ') {
            if (!started || prev_sp) continue;   /* trim leading, collapse runs */
            *w++ = ' '; prev_sp = true;
        } else {
            *w++ = (char)c; prev_sp = false; started = true;
        }
    }
    while (w > s && w[-1] == ' ') w--;            /* trim trailing */
    *w = '\0';
    if (maxw <= 1) return;
    size_t cols = 0; char *p = s;
    while (*p) {
        if (((unsigned char)*p & 0xC0) != 0x80) {
            if ((int)cols >= maxw - 1) break;     /* leave a column for "…" */
            cols++;
        }
        p++;
    }
    if (*p) strcpy(p, "\xe2\x80\xa6");            /* … at the UTF-8 boundary */
}

/* Display-only extractor: copy the value of top-level string key "key" out of a
 * JSON argument blob into `out` (light unescaping). Not a general parser — just
 * enough to pull a path/command/query for the activity line. */
static bool json_str_field(const char *json, const char *key, char *out, size_t n) {
    if (!json || !key || n == 0) return false;
    char pat[64];
    int pl = snprintf(pat, sizeof pat, "\"%s\"", key);
    if (pl < 0 || (size_t)pl >= sizeof pat) return false;
    const char *p = strstr(json, pat);
    if (!p) return false;
    p += pl;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p != ':') return false;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p != '"') return false;                  /* only string values */
    p++;
    size_t o = 0;
    while (*p && *p != '"' && o + 1 < n) {
        if (*p == '\\' && p[1]) {
            p++;
            char c = *p;
            if (c == 'n') c = '\n'; else if (c == 't') c = '\t';
            else if (c == 'r') c = '\r';
            out[o++] = c;
        } else {
            out[o++] = *p;
        }
        p++;
    }
    out[o] = '\0';
    return o > 0;
}

/* 256-colour bullet by tool category: blue = inspect, green = mutate,
 * amber = shell, teal = network, grey = other. */
static const char *tool_bullet_color(const char *tool) {
    if (!tool) return "\033[38;5;245m";
    if (!strcmp(tool,"read")||!strcmp(tool,"list")||!strcmp(tool,"ls")||
        !strcmp(tool,"grep")||!strcmp(tool,"search")||!strcmp(tool,"glob")||
        !strcmp(tool,"tree")||!strcmp(tool,"cat")||!strcmp(tool,"find"))
        return "\033[38;5;110m";
    if (!strcmp(tool,"edit")||!strcmp(tool,"write")||!strcmp(tool,"scaffold")||
        !strcmp(tool,"patch")||!strcmp(tool,"apply_patch")||!strcmp(tool,"plan_write"))
        return "\033[38;5;71m";
    if (!strcmp(tool,"bash")||!strcmp(tool,"shell")||!strcmp(tool,"run"))
        return "\033[38;5;179m";
    if (!strcmp(tool,"web")||!strcmp(tool,"web_search")||!strcmp(tool,"web_fetch")||
        !strcmp(tool,"deepsearch")||!strcmp(tool,"fetch")||!strcmp(tool,"search_web"))
        return "\033[38;5;108m";
    return "\033[38;5;245m";
}

/* Pull the single most relevant argument for `tool` from its JSON args. */
static void tool_display_arg(const char *tool, const char *args, char *out, size_t n) {
    out[0] = '\0';
    if (!args) return;
    const char *pref = "path";
    if (tool) {
        if (!strcmp(tool,"bash")||!strcmp(tool,"shell")||!strcmp(tool,"run")) pref = "command";
        else if (!strcmp(tool,"grep")||!strcmp(tool,"search")) pref = "pattern";
        else if (!strcmp(tool,"web")||!strcmp(tool,"web_search")||!strcmp(tool,"deepsearch")) pref = "query";
        else if (!strcmp(tool,"web_fetch")||!strcmp(tool,"fetch")) pref = "url";
    }
    if (json_str_field(args, pref, out, n) && out[0]) return;
    static const char *keys[] = {"path","file_path","file","command","cmd",
                                 "pattern","query","q","url","name","slug",NULL};
    for (int i = 0; keys[i]; i++)
        if (json_str_field(args, keys[i], out, n) && out[0]) return;
}

/* One-line summary of a tool result; returns true when it reads as an error
 * (so the caller can colour the sub-line red). */
static bool tool_result_summary(const char *result, char *out, size_t n) {
    if (!result || !*result) { snprintf(out, n, "(no output)"); return false; }
    bool err = (result[0] == 'E' || result[0] == 'e') &&
               strncmp(result + 1, "rror", 4) == 0;
    size_t first_len = strcspn(result, "\n");
    size_t lines = 0;
    for (const char *p = result; *p; p++) if (*p == '\n') lines++;
    if (result[strlen(result) - 1] != '\n') lines++;
    if (err || lines <= 1) {                      /* show the (first) line itself */
        size_t m = first_len < n - 1 ? first_len : n - 1;
        memcpy(out, result, m); out[m] = '\0';
        tidy_arg(out, 60);
        if (!err && !out[0]) snprintf(out, n, "ok");
        return err;
    }
    snprintf(out, n, "%zu lines", lines);
    return false;
}

/* Activity header: "● tool  arg" with a category-coloured bullet. Printed
 * immediately (before the tool runs) so a slow local model still shows life. */
static void print_tool_activity(const char *tool, const char *args) {
    char arg[256];
    tool_display_arg(tool, args, arg, sizeof arg);
    tidy_arg(arg, 60);
    printf("%s\xe2\x97\x8f\033[0m \033[38;5;252m%s\033[0m",
           tool_bullet_color(tool), tool ? tool : "?");
    if (arg[0]) printf("  \033[38;5;245m%s\033[0m", arg);
    printf("\n");
    fflush(stdout);
}

/* Same, for the legacy path where the call is a "tool args…" shell string:
 * first token is the tool (for the bullet colour), the rest is the argument. */
static void print_tool_activity_raw(const char *cmd) {
    if (!cmd) return;
    char tool[32]; size_t i = 0;
    while (cmd[i] && cmd[i] != ' ' && i + 1 < sizeof tool) { tool[i] = cmd[i]; i++; }
    tool[i] = '\0';
    const char *arg = cmd + i;
    while (*arg == ' ') arg++;
    char abuf[256];
    snprintf(abuf, sizeof abuf, "%s", arg);
    tidy_arg(abuf, 60);
    printf("%s\xe2\x97\x8f\033[0m \033[38;5;252m%s\033[0m",
           tool_bullet_color(tool), tool[0] ? tool : "?");
    if (abuf[0]) printf("  \033[38;5;245m%s\033[0m", abuf);
    printf("\n");
    fflush(stdout);
}

/* Result sub-line: "    └ <summary>" — dim by default, red on error, with a
 * "· trimmed Nk" note when the result was truncated before the model saw it. */
static void print_tool_result_line(const char *result, size_t dropped) {
    char summary[160];
    bool err = tool_result_summary(result, summary, sizeof summary);
    if (dropped) {
        size_t l = strlen(summary);
        char cbuf[24];
        if (dropped >= 1000) snprintf(cbuf, sizeof cbuf, "%zuk", (dropped + 500) / 1000);
        else                 snprintf(cbuf, sizeof cbuf, "%zu", dropped);
        snprintf(summary + l, sizeof summary - l, " \xc2\xb7 trimmed %s", cbuf);
    }
    printf("    \033[38;5;240m\xe2\x94\x94 \033[0m%s%s\033[0m\n",
           err ? "\033[38;5;167m" : "\033[38;5;240m", summary);
    fflush(stdout);
}

/* ── Startup banner ────────────────────────────────────────────────────
 * A compact rounded info box printed once the model + context are live, so it
 * can show the ACTUAL loaded context (post-OOM-retry) and GPU layer count.
 * Interactive TTY only — the load path already gates this off pipes/-p. */
static void banner_print_row(size_t inner, const char *text) {
    size_t k = 0; while (text[k] && text[k] != ' ') k++;   /* first token = label */
    size_t cw = 2 + disp_width(text);                      /* 2-space indent */
    size_t pad = inner > cw ? inner - cw : 0;
    printf("\033[38;5;240m\xe2\x94\x82\033[0m  "
           "\033[38;5;245m%.*s\033[38;5;110m%s",
           (int)k, text, text + k);
    for (size_t i = 0; i < pad; i++) putchar(' ');
    printf("\033[0m\033[38;5;240m\xe2\x94\x82\033[0m\n");
}

static void print_startup_banner(const char *model_path, int n_ctx, int n_gpu) {
    char model_tag[32];
    derive_model_tag(model_path, model_tag, sizeof model_tag);
    if (!model_tag[0]) snprintf(model_tag, sizeof model_tag, "(model)");

    char cwd[512];
    if (!getcwd(cwd, sizeof cwd)) snprintf(cwd, sizeof cwd, "?");
    const char *home = getenv("HOME");
    char cwddisp[520];
    size_t hl = home ? strlen(home) : 0;
    if (hl && strncmp(cwd, home, hl) == 0 && (cwd[hl] == '/' || cwd[hl] == '\0'))
        snprintf(cwddisp, sizeof cwddisp, "~%s", cwd + hl);
    else
        snprintf(cwddisp, sizeof cwddisp, "%s", cwd);
    tidy_arg(cwddisp, 60);

    char row_model[80], row_ctx[80], row_cwd[540];
    snprintf(row_model, sizeof row_model, "model   %s", model_tag);
    snprintf(row_ctx,   sizeof row_ctx,   "ctx     %d tokens    gpu   %d layers", n_ctx, n_gpu);
    snprintf(row_cwd,   sizeof row_cwd,   "cwd     %s", cwddisp);

    const char *title = "BASI \xc2\xb7 local coding agent";
    size_t rows_w = disp_width(row_model);
    if (disp_width(row_ctx) > rows_w) rows_w = disp_width(row_ctx);
    if (disp_width(row_cwd) > rows_w) rows_w = disp_width(row_cwd);
    size_t inner = rows_w + 2;                      /* 2-space left indent */
    if (inner < disp_width(title) + 4) inner = disp_width(title) + 4;
    if (inner < 34) inner = 34;
    if (inner > 76) inner = 76;

    /* top border: ╭─ TITLE ──…──╮ */
    printf("\n\033[38;5;240m\xe2\x95\xad\xe2\x94\x80 \033[38;5;110m%s "
           "\033[38;5;240m", title);
    /* sequence between the corners so far: "─"(1) + " "(1) + title + " "(1) */
    size_t used = disp_width(title) + 3;
    for (size_t i = used; i < inner; i++) printf("\xe2\x94\x80");
    printf("\xe2\x95\xae\033[0m\n");

    banner_print_row(inner, row_model);
    banner_print_row(inner, row_ctx);
    banner_print_row(inner, row_cwd);

    /* bottom border */
    printf("\033[38;5;240m\xe2\x95\xb0");
    for (size_t i = 0; i < inner; i++) printf("\xe2\x94\x80");
    printf("\xe2\x95\xaf\033[0m\n\n");
    fflush(stdout);
}

/* ── /model: switch the active model ───────────────────────────────────
 * Resolves the request (no arg → picker TUI; a readable .gguf path → that file;
 * else a case-insensitive substring over the scanned models), records it as the
 * new default, and switches by RE-EXECING the process. Re-exec is the robust
 * path: it rebuilds the whole model → context → sampler → tool-grammar →
 * template stack through the normal startup instead of a fragile in-place
 * teardown, and gives correct fresh-KV semantics. The live conversation rides
 * across via --resume, so the chat continues under the new model with no menus.
 * Returns (REPL continues) on cancel / no-match / same model / exec failure;
 * on success execv never returns. */
static bool ci_contains(const char *hay, const char *needle) {
    if (!needle || !*needle) return true;
    size_t nl = strlen(needle);
    for (const char *h = hay; *h; h++) {
        size_t i = 0;
        while (i < nl && h[i]) {
            char a = h[i], b = needle[i];
            if (a >= 'A' && a <= 'Z') a += 32;
            if (b >= 'A' && b <= 'Z') b += 32;
            if (a != b) break;
            i++;
        }
        if (i == nl) return true;
    }
    return false;
}

static void try_model_switch(const char *arg, char **argv, int argc,
                             const char *cur_model, int cur_ngl,
                             const char *session_path,
                             struct llama_context *ctx) {
    char *new_path = NULL;              /* set for a named / substring switch */
    int   new_ngl  = cur_ngl;
    bool  use_picker = false;

    if (!arg || !*arg) {
        use_picker = true;              /* defer the picker to the fresh child */
    } else if (access(arg, R_OK) == 0 && strstr(arg, ".gguf")) {
        new_path = strdup(arg);         /* direct path */
    } else {                            /* substring over scanned models */
        char **models = NULL;
        int n = basi_list_models(&models);
        int match = -1, matches = 0;
        for (int i = 0; i < n; i++) {
            const char *base = strrchr(models[i], '/');
            if (ci_contains(base ? base + 1 : models[i], arg)) { matches++; match = i; }
        }
        if (matches == 1) {
            new_path = strdup(models[match]);
        } else if (matches == 0) {
            printf("\033[31m[/model: no model matches '%s']\033[0m\n", arg);
        } else {
            printf("\033[33m[/model: '%s' matches %d models — be more specific:]\033[0m\n",
                   arg, matches);
            for (int i = 0; i < n; i++) {
                const char *base = strrchr(models[i], '/');
                base = base ? base + 1 : models[i];
                if (ci_contains(base, arg)) printf("    %s\n", base);
            }
        }
        for (int i = 0; i < n; i++) free(models[i]);
        free(models);
        if (!new_path) { fflush(stdout); return; }   /* no switch; statusbar intact */
    }

    /* Named switch: skip if it already resolves to the running model. */
    if (new_path) {
        char rp_new[PATH_MAX], rp_cur[PATH_MAX];
        const char *a = realpath(new_path, rp_new) ? rp_new : new_path;
        const char *b = (cur_model && realpath(cur_model, rp_cur)) ? rp_cur : cur_model;
        if (b && strcmp(a, b) == 0) {
            const char *base = strrchr(new_path, '/');
            printf("\033[90m[Already using %s]\033[0m\n", base ? base + 1 : new_path);
            free(new_path); fflush(stdout);
            return;
        }
        save_default_model(new_path, new_ngl, 0);    /* persist as new default */
    }

    /* Hand off via re-exec. Replacing this process image releases the current
       model's VRAM, so the child — whether it shows the picker (--pick) or loads
       a named model — sees the WHOLE GPU free, not a GPU still holding this
       model. The conversation rides across via --resume. */
    statusbar_disable();
    disable_raw_mode();
    printf("\033[?25h");                /* show cursor for the clean re-launch */

    char **nv = calloc((size_t)argc + 8, sizeof(char *));
    int k = 0;
    nv[k++] = argv[0];
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--pick")) continue;                 /* valueless; may re-add */
        if (!strcmp(a,"-m") || !strcmp(a,"-ngl") || !strcmp(a,"--ngl") ||
            !strcmp(a,"-c") || !strcmp(a,"--ctx") || !strcmp(a,"--resume") ||
            !strcmp(a,"-p") || !strcmp(a,"--prompt") || !strcmp(a,"--print") ||
            !strcmp(a,"--deepsearch") || !strcmp(a,"-ds")) { i++; continue; }  /* + value */
        nv[k++] = (char *)a;
    }
    char nglbuf[16];
    if (use_picker) {
        nv[k++] = "--pick";
        printf("\033[38;5;240m\xe2\x97\x8f opening model picker \xe2\x80\xa6\033[0m\n");
    } else {
        snprintf(nglbuf, sizeof nglbuf, "%d", new_ngl);
        nv[k++] = "-m";   nv[k++] = new_path;
        nv[k++] = "-ngl"; nv[k++] = nglbuf;
        const char *base = strrchr(new_path, '/');
        printf("\033[38;5;240m\xe2\x97\x8f switching model \xe2\x86\x92 %s \xe2\x80\xa6\033[0m\n",
               base ? base + 1 : new_path);
    }
    if (session_path) { nv[k++] = "--resume"; nv[k++] = (char *)session_path; }
    nv[k] = NULL;
    fflush(stdout);

    /* Server mode: execv() replaces the image WITHOUT running atexit(kill_srv), so
       the spawned llama-server would be orphaned and the re-exec'd process could
       not rebind :8181 (address in use → new server fails to start). Tear it down
       now; the child spawns a fresh server for the new model. No-op if not in
       server mode (g_srv_pid == 0). */
    kill_srv();

    execv("/proc/self/exe", nv);        /* Linux: re-run this binary */
    execv(argv[0], nv);                 /* fallback */

    perror("/model: exec");             /* only reached if exec failed */
    free(nv);
    free(new_path);
    { char tag[32]; derive_model_tag(cur_model, tag, sizeof tag); statusbar_enable(ctx, tag); }
    fflush(stdout);
}

/* ── Context reclamation (recent-window compaction) ────────────────────
 * Resync after any mutation of the message history. The delta-prompt scheme
 * feeds only formatted_buf+prev_len each turn, trusting the KV cache to hold the
 * prefix; once we rewrite messages[] that prefix is invalid, so we drop the
 * whole KV and zero prev_len — the next render+decode rebuilds it from the
 * (now smaller) history. Shared by /clear, deepsearch-return, and reclaim. */
static void kv_resync_full(struct llama_context *ctx, size_t *prev_len) {
    llama_memory_clear(llama_get_memory(ctx), true);
    *prev_len = 0;
}

/* Cheap token estimate for window selection (chars/4, like opencode). The
 * compaction TRIGGER uses the exact KV count; only the choice of which messages
 * to keep uses this estimate, so approximation is fine. */
static int est_msg_tokens(const char *s) {
    return s ? (int)(strlen(s) / 4) : 0;
}

/* The anchored summary lives as a user message at messages[1], wrapped so the
 * model treats it as historical context. The opening tag also lets reclaim find
 * and UPDATE it on the next compaction (rolling/anchored summary). Rendered as a
 * plain user turn — exactly how opencode emits its conversation checkpoint, so no
 * chat-template change is needed. */
#define CHECKPOINT_TAG "<conversation-checkpoint>"

static const char *SUMMARY_SYS =
    "You compress a conversation into a compact, faithful checkpoint. Output ONLY "
    "the Markdown structure asked for, sections in order, terse bullets. Preserve "
    "VERBATIM every concrete value: file paths, commands, identifiers, error strings, "
    "numbers, names, and ANYTHING the user explicitly asked you to remember (secrets, "
    "passphrases, keys, codes) — even a single such line buried in unrelated bulk text "
    "must be carried forward exactly. Do not add prose and do not mention that the "
    "conversation was summarized.";

static const char *SUMMARY_TEMPLATE =
    "Produce EXACTLY this Markdown (keep every section; write \"(none)\" when empty):\n"
    "## Goal\n- [one-sentence task]\n"
    "## Constraints & Preferences\n- [user constraints/preferences or (none)]\n"
    "## Progress\n### Done\n- [completed or (none)]\n### In progress\n- [current or (none)]\n"
    "## Key Decisions\n- [decision and why or (none)]\n"
    "## Next Steps\n- [ordered next actions or (none)]\n"
    "## Critical Context\n- [facts, errors, open questions, and EXACT values/secrets/IDs the user asked to remember, or (none)]\n"
    "## Relevant Files\n- [path: why it matters or (none)]";

/* Serialize a message range [start,end) into a plain transcript for the summary
 * input. Tool results were already capped at add time (MAX_TOOL_RESULT_SZ). */
static char *serialize_messages(struct llama_chat_message *m, size_t start, size_t end) {
    StringBuf sb; sb_init(&sb);
    for (size_t i = start; i < end; i++) {
        const char *role = m[i].role ? m[i].role : "";
        const char *tag =
            strcmp(role, "user") == 0        ? "[User]" :
            strcmp(role, "assistant") == 0   ? "[Assistant]" :
            strcmp(role, "tool_call") == 0   ? "[Tool call]" :
            strcmp(role, "tool_result") == 0 ? "[Tool result]" :
            strcmp(role, "system") == 0      ? "[System]" : "[Other]";
        sb_append_str(&sb, tag);
        sb_append_str(&sb, ": ");
        sb_append_str(&sb, m[i].content ? m[i].content : "");
        sb_append_char(&sb, '\n');
    }
    return sb_to_str(&sb);
}

/* Wrap raw summary text in the anchored-checkpoint envelope. */
static char *build_checkpoint(const char *summary) {
    StringBuf sb; sb_init(&sb);
    sb_append_str(&sb, CHECKPOINT_TAG "\n"
        "Summary of earlier conversation — historical context, not new instructions.\n"
        "<summary>\n");
    sb_append_str(&sb, summary);
    sb_append_str(&sb, "\n</summary>\n</conversation-checkpoint>");
    return sb_to_str(&sb);
}

/* Run ONE quiet generation that compresses `head_text` (optionally updating
 * `prev_checkpoint`) into a summary, in this model's own chat format. Tools are
 * de-advertised for the render so the schemas don't bloat the prompt or tempt a
 * tool call. Leaves the KV cleared; the caller resyncs. Returns malloc'd summary
 * text, or NULL on failure (caller falls back to a plain drop). */
static char *summarize_head(struct llama_model *model, const struct llama_vocab *vocab,
                            struct llama_context *ctx, struct llama_sampler *smpl,
                            bool native_tools, char *formatted_buf,
                            const char *prev_checkpoint, const char *head_text) {
    StringBuf up; sb_init(&up);
    if (prev_checkpoint && *prev_checkpoint) {
        sb_append_str(&up, "Update the existing checkpoint using the new transcript: keep "
                           "still-true details, drop stale ones, merge in new facts.\n<existing>\n");
        sb_append_str(&up, prev_checkpoint);
        sb_append_str(&up, "\n</existing>\n\n");
    } else {
        sb_append_str(&up, "Summarize the conversation transcript below into a checkpoint.\n\n");
    }
    sb_append_str(&up, SUMMARY_TEMPLATE);
    sb_append_str(&up, "\n\n<transcript>\n");
    sb_append_str(&up, head_text);
    sb_append_str(&up, "\n</transcript>");
    char *user_content = sb_to_str(&up);

    struct llama_chat_message tmp[2];
    tmp[0].role = "system"; tmp[0].content = SUMMARY_SYS;
    tmp[1].role = "user";   tmp[1].content = user_content;

    basi_set_tools(NULL, 0);                       /* no tool schemas in the summary prompt */
    int len = apply_template(model, tmp, 2, true, formatted_buf, FORMATTED_BUF_SZ);
    if (native_tools) { int n; const BasiToolDef *d = basi_tool_defs(&n); basi_set_tools(d, n); }
    free(user_content);
    if (len <= 0) return NULL;

    llama_memory_clear(llama_get_memory(ctx), true);   /* fresh KV for the summary decode */
    sig_atomic_t prev_quiet = generate_quiet;
    generate_quiet = 1;
    GenerateResult r = generate(ctx, vocab, smpl, formatted_buf, (size_t)len);
    generate_quiet = prev_quiet;
    return r.text;                                  /* may be an "[error]" string; caller checks */
}

/* Recent-window reclamation with an anchored rolling summary (phase 2). When the
 * KV is within RESERVE of full: pin the system prompt (messages[0]); summarize the
 * older turns into / merged with a checkpoint at messages[1]; keep the newest whole
 * turns verbatim; resync the KV so the next render re-decodes the compacted history
 * once. The summary is the one irreducible LLM step; trigger, window selection and
 * KV resync are deterministic. If summarization fails it falls back to a plain drop
 * (phase-1 behaviour) so the session still survives. Returns true if it compacted.
 * Must be called BEFORE rendering the current turn. */
static bool reclaim_context_if_needed(
        struct llama_model *model, const struct llama_vocab *vocab,
        struct llama_context *ctx, struct llama_sampler *smpl,
        bool native_tools, char *formatted_buf,
        struct llama_chat_message **messages_p, size_t *msg_count_p,
        size_t *prev_len_p) {
    int total = (int)llama_n_ctx(ctx);
    /* RESERVE: headroom left free for the incoming turn + its answer. ~4k on a
       full local ctx, scaled down so it never swallows a small ctx whole. */
    int reserve = total / 4 < 4096 ? total / 4 : 4096;
    int used = context_used_tokens(ctx);

    struct llama_chat_message *m = *messages_p;
    size_t mc = *msg_count_p;

    /* Account for the turn we are ABOUT to decode: the just-added user message
       (m[mc-1], not yet in the KV) plus an answer's worth of headroom (reserve).
       Triggering on `used` alone misses a single large incoming turn that would
       overflow the context mid-decode and hit the [Context limit reached] wall. */
    int incoming = mc > 0 ? est_msg_tokens(m[mc - 1].content) : 0;
    if (getenv("BASI_DEBUG_RECLAIM"))
        fprintf(stderr, "[reclaim] used=%d incoming=%d total=%d reserve=%d mc=%zu -> %s\n",
                used, incoming, total, reserve, mc,
                used + incoming > total - reserve ? "TRIGGER" : "skip");
    if (used + incoming <= total - reserve) return false;   /* room for the turn + an answer */

    /* An existing anchored summary at index 1 is pinned alongside the system
       prompt and fed back in as `prev_checkpoint` so the summary rolls forward. */
    bool has_summary = mc >= 2 && strcmp(m[1].role, "user") == 0 &&
                       strncmp(m[1].content, CHECKPOINT_TAG, strlen(CHECKPOINT_TAG)) == 0;
    size_t pinned = has_summary ? 2 : 1;                /* [0..pinned) never dropped */
    if (mc <= pinned + 1) return false;                 /* only one turn beyond pins */

    /* The TRIGGER counts exact KV tokens, but per-message est (chars/4) misses the
       system prompt's render-time tool schemas + chat-template markup — so est can
       say "it all fits" while the real KV is over. Recover that fixed overhead as
       (used − summed est) and size the recent-window budget in REAL tokens, low
       enough that the post-compaction re-decode lands well under the trigger
       (target = total − 2·reserve), leaving room for the summary itself. */
    int est_conv = 0;
    for (size_t i = pinned; i < mc; i++) est_conv += est_msg_tokens(m[i].content);
    int overhead = used - est_conv;
    if (overhead < 0) overhead = 0;
    int keep_budget = (total - 2 * reserve) - overhead - reserve / 2;
    if (keep_budget < 0) keep_budget = 0;
    size_t keep_from = mc;
    int est = 0;
    for (size_t i = mc; i-- > pinned; ) {
        est += est_msg_tokens(m[i].content);
        if (strcmp(m[i].role, "user") == 0) {
            if (est <= keep_budget) keep_from = i;
            else break;
        }
    }
    if (keep_from == mc) {                              /* newest turn alone overflows KEEP: keep it */
        for (size_t i = mc; i-- > pinned; )
            if (strcmp(m[i].role, "user") == 0) { keep_from = i; break; }
    }
    if (getenv("BASI_DEBUG_RECLAIM")) {
        fprintf(stderr, "[reclaim2] mc=%zu pinned=%zu has_summary=%d keep_from=%zu kb=%d roles=",
                mc, pinned, (int)has_summary, keep_from, keep_budget);
        for (size_t i = 0; i < mc; i++)
            fprintf(stderr, "%zu:%s(%d) ", i, m[i].role ? m[i].role : "?", est_msg_tokens(m[i].content));
        fprintf(stderr, "\n");
    }
    if (keep_from <= pinned) return false;              /* nothing older than the kept suffix */

    size_t dropped = keep_from - pinned;
    bool want_summary  = (compact_mode == COMPACT_SUMMARY  || compact_mode == COMPACT_HYBRID);
    bool want_retrieve = (compact_mode == COMPACT_RETRIEVE || compact_mode == COMPACT_HYBRID);

    printf("\033[33m[Compacting context: %zu older message%s, %d/%d tokens, mode=%s...]\033[0m\n",
           dropped, dropped == 1 ? "" : "s", used, total, compact_mode_name(compact_mode));
    fflush(stdout);

    /* retrieve/hybrid: embed the dropped turns into the session index (verbatim,
       model-agnostic) BEFORE freeing them. */
    if (want_retrieve)
        for (size_t i = pinned; i < keep_from; i++)
            mem_add(m[i].content);

    /* summary/hybrid: (re)build the anchored checkpoint via one quiet generation. */
    char *summary = NULL;
    bool ok = false;
    if (want_summary) {
        char *head_text = serialize_messages(m, pinned, keep_from);
        summary = summarize_head(model, vocab, ctx, smpl, native_tools, formatted_buf,
                                 has_summary ? m[1].content : NULL, head_text);
        free(head_text);
        if (getenv("BASI_DEBUG_RECLAIM"))
            fprintf(stderr, "[reclaim-summary]\n%s\n[/reclaim-summary]\n", summary ? summary : "(null)");
        ok = summary && summary[0] && summary[0] != '[';   /* guards "[Tokenization failed]" etc. */
    }

    for (size_t i = pinned; i < keep_from; i++) free((void *)m[i].content);   /* drop head */
    size_t R = mc - keep_from;                          /* recent messages kept verbatim */
    if (ok) {
        char *cp = build_checkpoint(summary);
        if (has_summary) free((void *)m[1].content);    /* replace the old summary in place */
        m[1].role = "user";
        m[1].content = cp;
        memmove(&m[2], &m[keep_from], R * sizeof(struct llama_chat_message));
        *msg_count_p = 2 + R;
    } else {
        /* No checkpoint (retrieve/off, or summary failed): keep system [+ any
           existing summary] + recent. */
        memmove(&m[pinned], &m[keep_from], R * sizeof(struct llama_chat_message));
        *msg_count_p = pinned + R;
    }
    free(summary);

    kv_resync_full(ctx, prev_len_p);
    printf("\033[33m[Compacted: kept %zu recent message%s%s%s]\033[0m\n",
           R, R == 1 ? "" : "s",
           ok ? ", summary anchored" : "",
           want_retrieve ? ", indexed for retrieval" : "");
    fflush(stdout);
    return true;
}

/* ── Main ──────────────────────────────────────────────────────────── */


/* ── Slash-command dispatch (interactive REPL) ───── */
/* Handles any input beginning with '/'. Takes ownership of user_input and
 * frees it. State the commands mutate is passed by pointer; the macro
 * aliases keep the moved body verbatim. */
static void handle_slash_command(char *user_input,
        struct llama_model *model, const struct llama_vocab *vocab,
        struct llama_context *ctx,
        struct llama_chat_message **messages_p, size_t *msg_count_p,
        size_t *msg_cap_p, FILE *session_fp, size_t *prev_len_p,
        size_t session_prompt_tokens, size_t session_gen_tokens) {
    #define messages  (*messages_p)
    #define msg_count (*msg_count_p)
    #define prev_len  (*prev_len_p)
    #define ADD_MESSAGE(role_str, content_str) \
        repl_add_message(messages_p, msg_count_p, msg_cap_p, session_fp, (role_str), (content_str))

            if (strcmp(user_input, "/help") == 0) {
                printf(
                    "\nSlash commands:\n"
                    "  /help                 this help\n"
                    "  /clear                drop conversation history (system prompt + project memory kept)\n"
                    "  /cost                 show session token usage\n"
                    "  /save [path]          export transcript as JSONL\n"
                    "  /memory               open ./BASI.md in $EDITOR\n"
                    "  /note <text>          append a one-line note to the project knowledge base\n"
                    "  /edit <path>          open a knowledge-base file in $EDITOR (path under .basi/knowledge/)\n"
                    "  /permissions [mode]   show or set permission mode (default | accept-edits | bypass)\n"
                    "  /plan [<slug>|accept|off]\n"
                    "                        no args: show phase. <slug>: enter drafting. accept: drafting/premortem -> active. off: exit.\n"
                    "  /premortem            (drafting only) enter premortem — model rewrites the plan with a ## Pre-mortem section\n"
                    "  /deepsearch <question>\n"
                    "                        multi-round deep research (web + knowledge base), synthesized + cited\n"
                    "  /model [name]         switch model (no arg: picker; name: match; keeps your chat)\n"
                    "  /cookbook [sub]       download & manage models (list | search | get <repo> | rm)\n"
                    "\n"
                    "Subcommands (run before model load):\n"
                    "  basi-cli docs add <file.md> [--shelf=notes|pinned|docs]\n"
                    "                        copy a markdown file into ./.basi/knowledge/\n"
                    "\n"
                    "Tools the model can call: read, head, tail, grep, wc, bash,\n"
                    "  edit, scaffold, web_search, web_fetch, readfile, code_context,\n"
                    "  docs_toc, docs_get, docs_search, docs_recent_notes,\n"
                    "  docs_vector_search,\n"
                    "  plan_write (drafting/premortem), assumptions (drafting),\n"
                    "  spike_write (spike), plan_verify (active).\n\n");
                fflush(stdout);
                free(user_input);
                return;
            }
            if (strcmp(user_input, "/clear") == 0) {
                for (size_t i = 1; i < msg_count; i++) free((void*)messages[i].content);
                msg_count = 1;
                kv_resync_full(ctx, prev_len_p);
                mem_clear();    /* drop retrieval memory too */
                printf("\033[90m[Cleared conversation history (system prompt kept)]\033[0m\n");
                fflush(stdout);
                free(user_input);
                return;
            }
            if (strcmp(user_input, "/cost") == 0) {
                printf("\033[90m[Session: %zu prompt tokens, %zu generated tokens, %zu total]\033[0m\n",
                       session_prompt_tokens, session_gen_tokens,
                       session_prompt_tokens + session_gen_tokens);
                int used  = context_used_tokens(ctx);
                int total = (int)llama_n_ctx(ctx);
                int pct   = total > 0 ? (int)((100.0 * used) / total) : 0;
                printf("\033[90m[Context: %d/%d tokens in window (%d%%), %d free]\033[0m\n",
                       used, total, pct, total - used);
                fflush(stdout);
                free(user_input);
                return;
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
                        StringBuf line;
                        sb_init(&line);
                        sb_append_str(&line, "{\"role\":");
                        json_escape_into(&line, messages[i].role);
                        sb_append_str(&line, ",\"content\":");
                        json_escape_into(&line, messages[i].content);
                        sb_append_str(&line, "}\n");
                        fwrite(line.data, 1, line.len, out);
                        sb_free(&line);
                    }
                    fclose(out);
                    printf("\033[90m[Saved %zu messages to %s]\033[0m\n", msg_count, path);
                }
                fflush(stdout);
                free(user_input);
                return;
            }
            if (strcmp(user_input, "/memory") == 0) {
                const char *editor = getenv("EDITOR");
                if (!editor || !*editor) editor = "vi";
                char cmd[256];
                snprintf(cmd, sizeof(cmd), "%s BASI.md", editor);
                bool was_raw = raw_mode_enabled;
                statusbar_suspend();          /* give the bottom row back to $EDITOR */
                if (was_raw) disable_raw_mode();
                int rc = system(cmd);
                if (was_raw) enable_raw_mode();
                statusbar_resume();
                if (rc != 0) printf("\033[31m[Editor exited with status %d]\033[0m\n", rc);
                printf("\033[90m[BASI.md edited. Changes apply on next BASI restart.]\033[0m\n");
                fflush(stdout);
                free(user_input);
                return;
            }
            if (strncmp(user_input, "/note", 5) == 0 &&
                (user_input[5] == '\0' || user_input[5] == ' ')) {
                const char *text = user_input + 5;
                while (*text == ' ') text++;
                if (!*text) {
                    printf("\033[31m[/note: missing text — usage: /note <one-line note>]\033[0m\n");
                    fflush(stdout);
                    free(user_input);
                    return;
                }
                if (kb_ensure_dirs() != 0) {
                    printf("\033[31m[/note: cannot create .basi/knowledge/ tree (%s)]\033[0m\n",
                           strerror(errno));
                    fflush(stdout);
                    free(user_input);
                    return;
                }
                time_t now = time(NULL);
                struct tm tm_now;
                localtime_r(&now, &tm_now);
                char date[16], hhmm[8];
                strftime(date, sizeof(date), "%Y-%m-%d", &tm_now);
                strftime(hhmm, sizeof(hhmm), "%H:%M", &tm_now);
                char notepath[512];
                snprintf(notepath, sizeof(notepath), "%s/session-%s.md", KB_NOTES_DIR, date);
                struct stat st;
                bool fresh = (stat(notepath, &st) != 0);
                FILE *nf = fopen(notepath, "a");
                if (!nf) {
                    printf("\033[31m[/note: cannot write %s (%s)]\033[0m\n",
                           notepath, strerror(errno));
                    fflush(stdout);
                    free(user_input);
                    return;
                }
                if (fresh) {
                    fprintf(nf,
                        "---\nshelf: notes\ntitle: Session notes %s\nfetched: %s\n---\n\n",
                        date, date);
                }
                fprintf(nf, "- %s %s\n", hhmm, text);
                fclose(nf);
                printf("\033[90m[note saved to %s]\033[0m\n", notepath);
                fflush(stdout);
                free(user_input);
                return;
            }
            if (strncmp(user_input, "/edit", 5) == 0 &&
                (user_input[5] == '\0' || user_input[5] == ' ')) {
                const char *arg = user_input + 5;
                while (*arg == ' ') arg++;
                if (!*arg) {
                    printf("\033[31m[/edit: missing path — usage: /edit <path-under-.basi/knowledge/>]\033[0m\n");
                    fflush(stdout);
                    free(user_input);
                    return;
                }
                if (strstr(arg, "..") != NULL || arg[0] == '/') {
                    printf("\033[31m[/edit not allowed: path must be relative under .basi/knowledge/ and may not contain '..']\033[0m\n");
                    fflush(stdout);
                    free(user_input);
                    return;
                }
                const char *editor = getenv("EDITOR");
                if (!editor || !*editor) {
                    printf("\033[31m[/edit not allowed: $EDITOR is not set]\033[0m\n");
                    fflush(stdout);
                    free(user_input);
                    return;
                }
                char fullpath[1280];
                snprintf(fullpath, sizeof(fullpath), "%s/%s", KB_KNOW_DIR, arg);
                struct stat est;
                if (stat(fullpath, &est) != 0) {
                    printf("\033[31m[/edit not allowed: file not found: %s]\033[0m\n", fullpath);
                    fflush(stdout);
                    free(user_input);
                    return;
                }
                char cmdbuf[1536];
                snprintf(cmdbuf, sizeof(cmdbuf), "%s %s", editor, fullpath);
                bool was_raw = raw_mode_enabled;
                statusbar_suspend();          /* give the bottom row back to $EDITOR */
                if (was_raw) disable_raw_mode();
                int rc = system(cmdbuf);
                if (was_raw) enable_raw_mode();
                statusbar_resume();
                if (rc != 0) printf("\033[31m[Editor exited with status %d]\033[0m\n", rc);
                else printf("\033[90m[%s edited]\033[0m\n", fullpath);
                fflush(stdout);
                free(user_input);
                return;
            }
            if (strncmp(user_input, "/cookbook", 9) == 0 &&
                (user_input[9] == '\0' || user_input[9] == ' ')) {
                const char *arg = user_input + 9;
                while (*arg == ' ') arg++;
                /* get/rm stream a progress bar and read a confirmation, so hand
                   the terminal back to cooked mode (mirror /edit's shell-out). */
                bool was_raw = raw_mode_enabled;
                statusbar_suspend();
                if (was_raw) disable_raw_mode();
                cookbook_command(arg);
                if (was_raw) enable_raw_mode();
                statusbar_resume();
                fflush(stdout);
                free(user_input);
                return;
            }
            /* /model is intercepted in the REPL loop (it re-execs), so it never
               reaches here. */
            if (strncmp(user_input, "/deepsearch", 11) == 0 &&
                (user_input[11] == '\0' || user_input[11] == ' ')) {
                const char *q = user_input + 11;
                while (*q == ' ') q++;
                if (!*q) {
                    printf("\033[31m[/deepsearch: needs a question. Usage: /deepsearch <your research question>]\033[0m\n");
                    fflush(stdout);
                    free(user_input);
                    return;
                }
                char *q_copy = strdup(q);
                char *answer = execute_deep_search(model, vocab, q_copy);
                printf("\n%s\n\n", answer ? answer : "(no answer)");
                fflush(stdout);
                /* Record the exchange so follow-up turns have context. The
                 * research ran in its own context, so the main KV cache is
                 * untouched and out of sync with these new messages — clear it
                 * (like /clear) so the next turn cleanly re-decodes everything. */
                ADD_MESSAGE("user", q_copy);
                ADD_MESSAGE("assistant", answer ? answer : "(no answer)");
                kv_resync_full(ctx, prev_len_p);
                free(q_copy);
                free(answer);
                free(user_input);
                return;
            }
            if (strncmp(user_input, "/plan", 5) == 0 &&
                (user_input[5] == '\0' || user_input[5] == ' ')) {
                const char *arg = user_input + 5;
                while (*arg == ' ') arg++;
                /* Reject anything past the first token. */
                const char *tail = arg;
                while (*tail && *tail != ' ' && *tail != '\t') tail++;
                const char *rest = tail;
                while (*rest == ' ' || *rest == '\t') rest++;
                if (*rest) {
                    printf("\033[31m[/plan: unexpected extra argument. Use /plan, /plan <slug>, /plan accept, or /plan off]\033[0m\n");
                    fflush(stdout);
                    free(user_input);
                    return;
                }

                if (!*arg) {
                    if (plan_phase == PHASE_NONE) {
                        printf("\033[90m[Plan phase: none. Enter drafting with /plan <slug>.]\033[0m\n");
                    } else {
                        printf("\033[36m[Plan phase: %s — slug '%s']\033[0m\n",
                               plan_phase_name(plan_phase),
                               current_plan_slug ? current_plan_slug : "(unset)");
                    }
                } else if (strcmp(arg, "off") == 0) {
                    if (plan_phase == PHASE_NONE) {
                        printf("\033[90m[Plan phase already off]\033[0m\n");
                    } else {
                        plan_phase = PHASE_NONE;
                        free(current_plan_slug);
                        current_plan_slug = NULL;
                        spike_cycles = 0;
                        spike_calls  = 0;
                        printf("\033[90m[Plan phase: off]\033[0m\n");
                    }
                } else if (strcmp(arg, "accept") == 0) {
                    if (plan_phase == PHASE_DRAFTING || plan_phase == PHASE_PREMORTEM) {
                        plan_phase = PHASE_ACTIVE;
                        if (current_plan_slug) {
                            int src = rewrite_plan_status(current_plan_slug, "active");
                            if (src == -2) {
                                printf("\033[33m[Plan file has no status: line; runtime phase=active but file untouched.]\033[0m\n");
                            } else if (src == -1) {
                                printf("\033[33m[Plan file not yet written; runtime phase=active. Status will be set on the next plan_write.]\033[0m\n");
                            }
                        }
                        printf("\033[36m[Plan accepted: '%s' — phase=active. Tools unblocked.]\033[0m\n",
                               current_plan_slug ? current_plan_slug : "(unset)");
                    } else {
                        printf("\033[31m[/plan accept: only valid from drafting or premortem (current: %s)]\033[0m\n",
                               plan_phase_name(plan_phase));
                    }
                } else {
                    /* Treat as slug → enter drafting. */
                    if (plan_phase != PHASE_NONE) {
                        printf("\033[31m[/plan %s: already in %s phase for '%s'. Use /plan off to exit first.]\033[0m\n",
                               arg, plan_phase_name(plan_phase),
                               current_plan_slug ? current_plan_slug : "(unset)");
                    } else if (!kb_slug_valid(arg)) {
                        printf("\033[31m[/plan %s: invalid slug. Use lowercase a-z, 0-9, single hyphens; must start with a letter; max 64 chars.]\033[0m\n",
                               arg);
                    } else {
                        plan_phase = PHASE_DRAFTING;
                        free(current_plan_slug);
                        current_plan_slug = strdup(arg);
                        spike_cycles = 0;
                        spike_calls  = 0;
                        printf("\033[36m[Plan phase: drafting — slug '%s'. Research first, then save with plan_write. /plan accept when ready.]\033[0m\n",
                               current_plan_slug);
                    }
                }
                fflush(stdout);
                free(user_input);
                return;
            }
            if (strcmp(user_input, "/premortem") == 0) {
                if (plan_phase != PHASE_DRAFTING) {
                    printf("\033[31m[/premortem: only valid from drafting phase (current: %s)]\033[0m\n",
                           plan_phase_name(plan_phase));
                } else if (!current_plan_slug) {
                    printf("\033[31m[/premortem: no current plan slug — internal state error]\033[0m\n");
                } else {
                    char path[1024];
                    snprintf(path, sizeof(path), "%s/%s.md", KB_PLANS_DIR, current_plan_slug);
                    struct stat st;
                    if (stat(path, &st) != 0) {
                        printf("\033[31m[/premortem: no plan file at %s — model hasn't called plan_write yet. Have it draft the plan first.]\033[0m\n",
                               path);
                    } else {
                        plan_phase = PHASE_PREMORTEM;
                        int src = rewrite_plan_status(current_plan_slug, "premortem");
                        if (src == -2) {
                            printf("\033[33m[Plan file has no status: line; runtime phase=premortem but file untouched.]\033[0m\n");
                        }
                        printf("\033[36m[Pre-mortem phase entered for '%s'. On your next message the model will run Klein's protocol — \"imagine the plan failed; explain why\" — and rewrite %s with a ## Pre-mortem section. Use /plan accept when satisfied, or /plan off to abort.]\033[0m\n",
                               current_plan_slug, path);
                    }
                }
                fflush(stdout);
                free(user_input);
                return;
            }
            if (strncmp(user_input, "/permissions", 12) == 0 &&
                (user_input[12] == '\0' || user_input[12] == ' ')) {
                const char *arg = user_input + 12;
                while (*arg == ' ') arg++;
                if (!*arg) {
                    printf("\033[90m[Permission mode: %s]\033[0m\n", perm_mode_name(permission_mode));
                    printf("Modes:\n"
                           "  default       prompt before bash, edit, scaffold\n"
                           "  accept-edits  auto-approve edit + scaffold; bash still prompts\n"
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
                return;
            }
            printf("\033[33m[Unknown command '%s'. Type /help for the list.]\033[0m\n", user_input);
            fflush(stdout);
            free(user_input);
            return;
            #undef messages
    #undef msg_count
    #undef prev_len
    #undef ADD_MESSAGE
}

/* ── One agentic turn (generate → tool loop → answer) ───── */
/* Runs a single user turn: render the prompt, generate, dispatch any tool
 * call and feed the result back (up to max_tool_iterations), then a final
 * answer. State that outlives the turn is passed by pointer; user_input is
 * borrowed (the caller frees it). */
static void run_agentic_turn(char *user_input,
        struct llama_model *model, const struct llama_vocab *vocab,
        struct llama_context *ctx, struct llama_sampler *smpl,
        bool native_tools, char *formatted_buf,
        struct llama_chat_message **messages_p, size_t *msg_count_p,
        size_t *msg_cap_p, FILE *session_fp, size_t *prev_len_p,
        size_t *session_prompt_tokens_p, size_t *session_gen_tokens_p) {
    #define messages  (*messages_p)
    #define msg_count (*msg_count_p)
    #define prev_len  (*prev_len_p)
    #define session_prompt_tokens (*session_prompt_tokens_p)
    #define session_gen_tokens    (*session_gen_tokens_p)
    #define ADD_MESSAGE(role_str, content_str) \
        repl_add_message(messages_p, msg_count_p, msg_cap_p, session_fp, (role_str), (content_str))
        /* Recompute the date every turn and ride it in with the user message.
           The system prompt is decoded once and cached, so only tokens added now
           reach the model — this keeps "today" correct on sessions left open for
           days or resumed weeks later. */
        char today[16];
        {
            time_t now = time(NULL);
            struct tm *t = localtime(&now);
            strftime(today, sizeof(today), "%Y-%m-%d", t);
        }

        const char *banner = plan_phase_banner(plan_phase);
        {
            size_t blen = strlen(user_input) + (banner ? strlen(banner) : 0) + 64;
            char *msg = malloc(blen);
            if (banner)
                snprintf(msg, blen, "[Today's date: %s]\n%s\n\n%s", today, banner, user_input);
            else
                snprintf(msg, blen, "[Today's date: %s]\n%s", today, user_input);
            ADD_MESSAGE("user", msg);
            free(msg);
        }
        /* keep user_input alive across tool iterations so the context reminder
         * can echo the original request back to the model — small models drift
         * after several tool rounds otherwise. Freed at end of turn. */

        /* Reclaim context if the KV is near full, BEFORE rendering this turn.
           Summarizes the oldest turns into an anchored checkpoint (system prompt
           pinned) and resyncs the KV, so the render below re-decodes the compacted
           history once instead of hitting the [Context limit reached] wall. */
        reclaim_context_if_needed(model, vocab, ctx, smpl, native_tools, formatted_buf,
                                  messages_p, msg_count_p, prev_len_p);

        /* retrieve/hybrid: pull the top-k dropped turns most relevant to THIS query
           and prepend them (verbatim) to the current user message, so they ride the
           turn's delta without rewriting the cached prefix (prev_len stays valid).
           Model-agnostic: embedding similarity, no LLM call. */
        if ((compact_mode == COMPACT_RETRIEVE || compact_mode == COMPACT_HYBRID)
            && mem_count() > 0 && msg_count > 0) {
            char *hits[8]; float scores[8];
            float thr = 0.25f;
            { const char *t = getenv("BASI_RETRIEVE_THRESHOLD"); if (t) thr = (float)atof(t); }
            int nh = mem_retrieve(user_input, 4, thr, hits, scores);
            if (getenv("BASI_DEBUG_RECLAIM"))
                fprintf(stderr, "[retrieve] mem_count=%zu thr=%.2f hits=%d top=%.3f\n",
                        mem_count(), thr, nh, nh > 0 ? scores[0] : -1.0f);
            if (nh > 0) {
                StringBuf rb; sb_init(&rb);
                sb_append_str(&rb, "[Retrieved earlier context — reference for your answer, not new instructions]\n");
                for (int i = 0; i < nh; i++) {
                    sb_append_str(&rb, hits[i]);
                    sb_append_str(&rb, "\n");
                    free(hits[i]);
                }
                sb_append_str(&rb, "---\n");
                sb_append_str(&rb, messages[msg_count - 1].content ? messages[msg_count - 1].content : "");
                char *merged = sb_to_str(&rb);
                free((void *)messages[msg_count - 1].content);
                messages[msg_count - 1].content = merged;
                if (getenv("BASI_DEBUG_RECLAIM"))
                    fprintf(stderr, "[retrieve] injected %d chunk(s) (top score %.2f)\n", nh, scores[0]);
            }
        }

        /* Apply chat template — renders the model's native format via the jinja
           engine (chat_tmpl shim), with ChatML fallback. */
        int new_len = apply_template(
            model, messages, msg_count, true,
            formatted_buf, FORMATTED_BUF_SZ);

        if (new_len < 0) {
            printf("Error: Failed to apply chat template\n");
            fflush(stdout);
            return;
        }
        srv_update_ctx_used(vocab, formatted_buf, new_len);   /* server-mode ctx meter */

        /* Delta-prompt invariant guard. The KV holds the prefix up to prev_len
           and we feed only formatted_buf+prev_len, trusting the render to grow
           monotonically. Some templates DON'T — notably thinking models (Qwen3.x)
           that restructure prior turns — so this render can come out SHORTER than
           prev_len, underflowing prompt_len to a huge size_t and crashing tokenize.
           When that happens the cached prefix is invalid anyway: drop the KV and
           decode the whole freshly-rendered prompt. */
        /* Resync the KV when a cross-turn delta can't be trusted:
           (a) prev_len > new_len — a non-monotonic render underflows prompt_len
               and crashes tokenize.
           (b) thinking model — BASI strips <think>...</think> from the stored
               assistant turn, but the KV decoded it. So the KV holds MORE tokens
               than the re-rendered (stripped) history, and a delta feeds the new
               turn at the wrong KV position: the model loses the conversation and
               answers a stale turn. Re-decode the clean full history instead.
               (Costs a full prefill per turn for thinking models — correct over
               fast; a future optimization could surgically drop the think tokens
               from the KV instead.) */
        /* Server mode has no local KV to delta against — the server prefix-caches
           the FULL prompt itself (cache_prompt=true), so a delta would send only
           the tail as the entire prompt and lose all history. Always feed full. */
        bool delta_unsafe = (prev_len > (size_t)new_len) || basi_thinking_tags(NULL, NULL) || basi_srv_port > 0;
        if (delta_unsafe) {
            if (getenv("BASI_DEBUG_RECLAIM"))
                fprintf(stderr, "[delta] resync (prev_len=%zu new_len=%d thinking=%d)\n",
                        prev_len, new_len, basi_thinking_tags(NULL, NULL));
            kv_resync_full(ctx, prev_len_p);
        }
        char *prompt = formatted_buf + prev_len;
        size_t prompt_len = (size_t)new_len - prev_len;

        /* Tool execution loop. The per-turn cap defaults to 40 — multi-step work
           on a large/unfamiliar repo (navigate→grep→read→edit→test→fix) needs many
           calls just to locate the file before it can edit; the old 15 starved
           real SWE-bench-scale tasks (the loop hit the cap mid-exploration, never
           editing). Overridable via BASI_MAX_TOOL_ITERS. */
        int tool_iterations = 0;
        int consec_parse_fail = 0;     /* consecutive unparseable tool-call outputs */
        int max_tool_iterations = 40;
        {
            const char *mi = getenv("BASI_MAX_TOOL_ITERS");
            if (mi) { int v = atoi(mi); if (v > 0 && v <= 200) max_tool_iterations = v; }
        }

        while (tool_iterations < max_tool_iterations) {
            tool_iterations++;

            /* Reclaim INSIDE the tool loop too. A long agentic turn is one user
               turn with many tool calls, each appending a result — with the
               reclaim check only before the loop, that history accumulates with
               no compaction until the KV overflows mid-session (the [Context
               limit reached] wall, after which a small model often degenerates).
               On compaction the KV is cleared and prev_len reset, so re-render the
               full compacted history instead of the now-invalid delta. Skip the
               first iteration: the pre-loop reclaim + retrieve injection already
               set up `prompt` for it. */
            if (tool_iterations > 1 &&
                reclaim_context_if_needed(model, vocab, ctx, smpl, native_tools,
                                          formatted_buf, messages_p, msg_count_p, prev_len_p)) {
                int rl = apply_template(model, messages, msg_count, true,
                                        formatted_buf, FORMATTED_BUF_SZ);
                if (rl < 0) { printf("Error: Failed to apply chat template\n"); fflush(stdout); break; }
                prompt = formatted_buf;
                prompt_len = (size_t)rl;
            }

            /* Reset the lazy tool grammar so this generation starts fresh — no
               trigger state carried over from the previous tool round. */
            if (g_tool_grammar) llama_sampler_reset(g_tool_grammar);
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
            char meter[80];
            format_context_meter(ctx, meter, sizeof meter);
            printf("\033[90m[ Prompt: %.1f t/s | Generation: %.1f t/s | %s ]\033[0m\n",
                   prompt_tps, gen_tps, meter);
            fflush(stdout);
            statusbar_draw();   /* refresh the pinned ctx meter after this turn */

            /* Detect a tool call: native function-calling if the model's
               template supports it, else the legacy <tool> tag. Both resolve
               to a command string that execute_tool() runs (reusing all
               dispatch + plan-phase gating). */
            char *cmd_str = NULL;          /* legacy <tool> path: command for execute_tool */
            char *unknown_tool = NULL;     /* malloc'd name if a native call hit an unknown tool */
            char *tool_result = NULL;      /* native path computes this directly */
            char *call_env = NULL;         /* native: structured assistant tool-call envelope */
            char *call_name = NULL;        /* native: tool name, for the result envelope */
            bool have_call = false;
            BasiToolCall *ncalls = NULL;
            int n_ncalls = 0;

            if (native_tools) {
                n_ncalls = basi_parse_tool_calls(result.text, &ncalls);
                if (n_ncalls > 0) {        /* one tool per turn for now (2a) */
                    const char *nm = ncalls[0].name ? ncalls[0].name : "?";
                    print_tool_activity(nm, ncalls[0].arguments);
                    /* Structured dispatch: parsed JSON args go straight to the
                       handlers, never round-tripped through a shell string. */
                    tool_result = execute_tool_native(ncalls[0].name, ncalls[0].arguments);
                    if (getenv("BASI_DEBUG_TOOLS"))
                        fprintf(stderr, "[tool] name=%s args=%s\n  -> result=%.160s\n",
                                nm, ncalls[0].arguments ? ncalls[0].arguments : "(null)",
                                tool_result ? tool_result : "(null)");
                    if (!tool_result) unknown_tool = strdup(nm);
                    call_name = strdup(nm);
                    call_env  = tool_call_envelope(ncalls[0].name, ncalls[0].arguments);
                    have_call = true;
                }
            } else {
                size_t tool_cmd_len;
                const char *tc = extract_tool_call(result.text, &tool_cmd_len);
                if (tc) {
                    cmd_str = malloc(tool_cmd_len + 1);
                    memcpy(cmd_str, tc, tool_cmd_len);
                    cmd_str[tool_cmd_len] = '\0';
                    have_call = true;
                }
            }

            if (have_call) {
                consec_parse_fail = 0;         /* parseable call — reset the retry counter */
                if (cmd_str) {                 /* legacy path: build → execute */
                    print_tool_activity_raw(cmd_str);
                    tool_result = execute_tool(cmd_str);
                    free(cmd_str);
                } else if (!tool_result) {     /* native path hit an unknown tool */
                    tool_result = malloc(256);
                    snprintf(tool_result, 256,
                        "Error: unknown tool '%s' — it is not one of the available functions.", unknown_tool);
                    /* activity header already shown; the red result sub-line reports it */
                }
                free(unknown_tool);
                if (ncalls) basi_free_tool_calls(ncalls, n_ncalls);

                /* Truncate tool result if too large — line-aware, head+tail.
                   The dim "└ <summary>" sub-line reports the outcome (and any
                   trim) directly under the activity header. */
                size_t dropped = truncate_tool_result(tool_result,
                        TOOL_RESULT_HEAD_LINES, TOOL_RESULT_TAIL_LINES,
                        TOOL_RESULT_MAX_BYTES);
                print_tool_result_line(tool_result, dropped);

                /* context_used_tokens() reads the server's tracked prompt count in
                   server mode, or the local KV otherwise — so the budget hint the
                   model sees is honest in both. */
                int used = context_used_tokens(ctx);
                int remaining = (int)llama_n_ctx(ctx) - used;

                if (native_tools) {
                    /* Structured round-trip: the assistant's tool call and the
                       result are stored as structured turns (decoded by the
                       chat_tmpl shim into common_chat tool_calls / a tool
                       message), so EVERY model's template renders the call and
                       its response — a bare role:"tool" content message is
                       dropped by some templates (e.g. Gemma). */
                    StringBuf tr;
                    sb_init(&tr);
                    sb_append_str(&tr, tool_result);
                    char budget[256];
                    snprintf(budget, sizeof(budget),
                        "\n[Context: %d/%d tokens used, %d remaining. Answer now if remaining < 8000.]",
                        used, (int)llama_n_ctx(ctx), remaining);
                    sb_append_str(&tr, budget);
                    char *res_env = tool_result_envelope(call_name, sb_to_str(&tr));
                    sb_free(&tr);
                    ADD_MESSAGE("tool_call", call_env);     /* assistant: the call */
                    ADD_MESSAGE("tool_result", res_env);    /* tool: the result   */
                    free(res_env);
                    free(result.text);
                } else {
                    /* Legacy <tool> path: raw assistant text + a user-wrapped
                       <tool_result> block (unchanged). */
                    ADD_MESSAGE("assistant", result.text);
                    free(result.text);
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
                }
                free(call_env);     /* native built it; NULL (safe) on legacy */
                free(call_name);
                free(tool_result);

                /* Update template for next iteration (delta prompt) */
                int next_len = apply_template(
                    model, messages, msg_count, true,
                    formatted_buf, FORMATTED_BUF_SZ);
                if (next_len < 0) {
                    printf("Error: Failed to apply chat template\n");
                    fflush(stdout);
                    break;
                }
                srv_update_ctx_used(vocab, formatted_buf, next_len);   /* server-mode ctx meter */
                int prev = apply_template(
                    model, messages, msg_count - 1, false, NULL, 0);
                /* Same non-monotonic-render guard as the turn-start delta: if the
                   all-but-last render is longer than the full one, the delta would
                   underflow — resync the KV and feed the whole prompt instead. */
                if (prev < 0 || prev > next_len || basi_srv_port > 0) {   /* server: always full prompt */
                    kv_resync_full(ctx, prev_len_p);
                    prev = 0;
                }
                prompt = formatted_buf + prev;
                prompt_len = (size_t)next_len - (size_t)prev;

                printf("\n");
                fflush(stdout);
            } else {
                if (ncalls) basi_free_tool_calls(ncalls, n_ncalls);
                /* Distinguish a genuine final answer from a MALFORMED tool call
                   (tool-format degeneration, e.g. <function=read>…</parameter></parameter>
                   that common_chat_peg_parse rejects). If the output shows a tool-call
                   attempt, don't end the turn — nudge the model to re-emit a valid call
                   and keep looping. Bounded by a small consecutive-failure cap so a
                   persistently-degenerate model still terminates. */
                int looks_like_call = native_tools && result.text &&
                    (strstr(result.text, "<tool_call>") ||
                     strstr(result.text, "<function=") ||
                     strstr(result.text, "<tool>"));
                if (looks_like_call && ++consec_parse_fail <= 3 &&
                    tool_iterations < max_tool_iterations) {
                    printf("\033[33m[Tool call unparseable — re-emit requested (%d/3)]\033[0m\n",
                           consec_parse_fail);
                    fflush(stdout);
                    ADD_MESSAGE("assistant", result.text);
                    ADD_MESSAGE("user",
                        "Your last message looked like a tool call but could not be parsed "
                        "(malformed syntax / mismatched tags). Re-emit exactly ONE tool call in "
                        "the correct format with properly matched tags, or give your final answer "
                        "if the task is complete.");
                    free(result.text);
                    int nl = apply_template(model, messages, msg_count, true,
                                            formatted_buf, FORMATTED_BUF_SZ);
                    if (nl < 0) { printf("Error: Failed to apply chat template\n"); break; }
                    int pv = apply_template(model, messages, msg_count - 1, false, NULL, 0);
                    if (pv < 0 || pv > nl || basi_srv_port > 0) { kv_resync_full(ctx, prev_len_p); pv = 0; }
                    prompt = formatted_buf + pv;
                    prompt_len = (size_t)nl - (size_t)pv;
                    printf("\n");
                    fflush(stdout);
                    continue;
                }
                /* genuine final answer (or gave up after repeated parse failures) */
                ADD_MESSAGE("assistant", result.text);
                free(result.text);
                break;
            }
        }

        /* If we exhausted tool iterations, do one final generation for the answer */
        if (tool_iterations >= max_tool_iterations) {
            printf("\033[90m[Generating answer...]\033[0m\n");
            fflush(stdout);
            if (g_tool_grammar) llama_sampler_reset(g_tool_grammar);
            generation_interrupted = 0;
            setup_sigint_handler();
            GenerateResult final_result = generate(ctx, vocab, smpl, prompt, prompt_len);
            reset_sigint_handler();
            session_prompt_tokens += final_result.prompt_tokens;
            session_gen_tokens    += final_result.gen_tokens;

            double gen_tps = final_result.gen_time_s > 0
                ? final_result.gen_tokens / final_result.gen_time_s : 0;
            char meter[80];
            format_context_meter(ctx, meter, sizeof meter);
            printf("\033[90m[ Generation: %.1f t/s | %s ]\033[0m\n", gen_tps, meter);
            fflush(stdout);
            statusbar_draw();   /* refresh the pinned ctx meter after this turn */

            ADD_MESSAGE("assistant", final_result.text);
            free(final_result.text);
        }

        printf("\n");
        fflush(stdout);

        /* Update prev_len for next turn */
        int len = apply_template(
            model, messages, msg_count, false, NULL, 0);
        if (len >= 0) prev_len = (size_t)len;

    #undef messages
    #undef msg_count
    #undef prev_len
    #undef session_prompt_tokens
    #undef session_gen_tokens
    #undef ADD_MESSAGE
}

int main(int argc, char **argv) {
    /* `basi-cli docs ...` subcommand: handle and exit before model load. */
    if (argc >= 2 && strcmp(argv[1], "docs") == 0) {
        if (argc >= 3 && strcmp(argv[2], "add") == 0) {
            return cmd_docs_add(argc, argv);
        }
        fprintf(stderr,
            "Usage:\n"
            "  basi-cli docs add <file.md> [--shelf=notes|pinned|docs]\n");
        return 2;
    }

    Cli cli = parse_args(argc, argv);
    if (cli.want_exit) return cli.exit_code;
    const char *model_path           = cli.model_path;
    int         n_gpu_layers         = cli.n_gpu_layers;
    bool        ngl_set              = cli.ngl_set;
    const char *oneshot_deepsearch_q = cli.deepsearch_q;
    const char *oneshot_prompt       = cli.prompt;
    bool        no_tools             = cli.no_tools;
    const char *system_override      = cli.system_override;
    int         cli_ctx              = cli.cli_ctx;
    float       cli_temp             = cli.cli_temp;
    uint32_t    cli_seed             = cli.cli_seed;
    /* top-k / top-p: CLI wins, else BASI_TOP_K / BASI_TOP_P env, else disabled
       (0 / 1.0). Some models — e.g. Qwythos-9B — recommend top_k 20 + top_p 0.95
       and warn that disabling the tail (or very low temp) triggers repeat loops. */
    int   top_k = cli.cli_top_k;
    float top_p = cli.cli_top_p;
    { const char *e = getenv("BASI_TOP_K"); if (e && cli.cli_top_k == 0) top_k = atoi(e); }
    { const char *e = getenv("BASI_TOP_P"); if (e && cli.cli_top_p == 1.0f) top_p = (float)atof(e); }
    const char *resume_path          = cli.resume_path;

    /* --yolo/--bypass: auto-approve every tool action. Without it, a
     * non-interactive -p run that triggers an approval prompt reads EOF on
     * stdin and the tool is denied (request_approval returns 0), so tools
     * silently never run. PERM_BYPASS short-circuits the approval at every
     * call site (bash/edit/scaffold) before the prompt is reached. */
    if (cli.bypass) permission_mode = PERM_BYPASS;

    /* Context-compaction strategy (Phase 4 A/B): BASI_COMPACT=off|summary|retrieve|hybrid. */
    {
        const char *cm = getenv("BASI_COMPACT");
        if (cm) {
            if      (strcmp(cm, "off") == 0)      compact_mode = COMPACT_OFF;
            else if (strcmp(cm, "summary") == 0)  compact_mode = COMPACT_SUMMARY;
            else if (strcmp(cm, "retrieve") == 0) compact_mode = COMPACT_RETRIEVE;
            else if (strcmp(cm, "hybrid") == 0)   compact_mode = COMPACT_HYBRID;
            else fprintf(stderr, "[warn] unknown BASI_COMPACT='%s'; using summary\n", cm);
        }
    }
    /* retrieve/hybrid need the embedder. Check it's findable up front (no load) and
       fall back to summary with a LOUD warning rather than silently dropping turns
       with no memory — important now that retrieve is the default. */
    if ((compact_mode == COMPACT_RETRIEVE || compact_mode == COMPACT_HYBRID)
        && !embed_available()) {
        fprintf(stderr,
            "\033[33m[warn] compaction '%s' needs an embedding model, but none was found:\n"
            "       %s\n"
            "       falling back to 'summary'. Set BASI_EMBED_MODEL to enable retrieval.\033[0m\n",
            compact_mode_name(compact_mode), embed_last_error());
        compact_mode = COMPACT_SUMMARY;
    }

    bool oneshot = (oneshot_deepsearch_q != NULL) || (oneshot_prompt != NULL);

    /* --no-tools is a modifier on -p: it only makes sense for the one-shot
     * prompt path. Reject it standalone rather than silently ignoring it. */
    if (no_tools && !oneshot_prompt) {
        fprintf(stderr,
            "Error: --no-tools only applies to -p/--prompt. "
            "Use: basi-cli -m <model> -p \"<prompt>\" --no-tools\n");
        return 1;
    }

    /* Model resolution order (interactive): -m (this invocation) > saved
       default (set by the picker / /model) > $BASI_MODEL > first-run picker.
       The saved default is what drops later launches straight into chat and
       what lets a /model choice persist. */
    static char picked_model[1024];
    static char default_model[1024];
    int ctx_override = 0;
    float temp_override = -1;
    int picker_spec = -1, picker_fa = -1;   /* server launch flags from the picker (-1 = not set) */
    bool loaded_from_default = false;   /* model came from the saved-default file */
    /* --pick (from /model): force the picker BEFORE any model is loaded, so its
       VRAM probe / auto-fit see the whole GPU free (the previous model was
       released when /model re-execed into this fresh process). A cancel falls
       through to the saved default below — i.e. reloads the same model. */
    if (cli.pick && !oneshot && !model_path) {
        LaunchConfig cfg = pick_model();
        if (cfg.model_path) {
            strncpy(picked_model, cfg.model_path, sizeof(picked_model) - 1);
            picked_model[sizeof(picked_model) - 1] = '\0';
            free(cfg.model_path);
            model_path = picked_model;
            if (!ngl_set) n_gpu_layers = cfg.gpu_layers;
            ctx_override  = cfg.ctx_size;
            temp_override = cfg.temperature;
            picker_spec   = cfg.spec_draft_mtp;
            picker_fa     = cfg.flash_attn;
            save_default_model(model_path, n_gpu_layers, cfg.ctx_size);
        }
    }
    if (!model_path) {
        int d_ngl = -1, d_ctx = 0;
        if (load_default_model(default_model, sizeof default_model, &d_ngl, &d_ctx)) {
            model_path = default_model;
            loaded_from_default = true;
            if (!ngl_set && d_ngl >= 0) n_gpu_layers = d_ngl;
            if (d_ctx > 0) ctx_override = d_ctx;
        }
    }
    if (!model_path) model_path = getenv("BASI_MODEL");
    if (!model_path && oneshot) {
        fprintf(stderr,
            "Error: non-interactive mode (--deepsearch / -p) needs a model — "
            "pass -m <model.gguf> or set BASI_MODEL.\n");
        return 1;
    }
    if (!model_path) {
        LaunchConfig cfg = pick_model();
        if (!cfg.model_path) {
            fprintf(stderr, "No model selected.\n");
            return 1;
        }
        strncpy(picked_model, cfg.model_path, sizeof(picked_model) - 1);
        picked_model[sizeof(picked_model) - 1] = '\0';  /* strncpy may not NUL-terminate */
        free(cfg.model_path);
        model_path = picked_model;
        if (!ngl_set) n_gpu_layers = cfg.gpu_layers;
        ctx_override = cfg.ctx_size;
        temp_override = cfg.temperature;
        picker_spec  = cfg.spec_draft_mtp;
        picker_fa    = cfg.flash_attn;
        /* An explicit pick always (re)writes the default — including repairing a
           stale file that pointed at a since-deleted model. */
        save_default_model(model_path, n_gpu_layers, cfg.ctx_size);
    }

    /* Explicit CLI knobs win over picker / built-in defaults. */
    if (cli_ctx  > 0)   ctx_override  = cli_ctx;
    if (cli_temp >= 0)  temp_override = cli_temp;

    /* Persist the model actually being loaded as the default, so the NEXT launch
       drops straight into chat instead of the picker. Covers the picker pick,
       -m, and $BASI_MODEL uniformly. Seed only when no default file exists yet,
       so a one-off `-m other.gguf` never overwrites a default the user chose (via
       the picker or /model); those paths write the file explicitly elsewhere. */
    if (!oneshot && model_path && !loaded_from_default && !default_model_file_exists())
        save_default_model(model_path, n_gpu_layers, ctx_override);

    /* Warm up the local SearXNG (web_search backend) while the model loads.
     * --no-tools never touches the web, so don't spin SearXNG up for it. */
    if (!no_tools) web_ensure_searxng();

    /* Server-backed generation spike (M1): BASI_SERVER_SELFTEST=1 spawns a
       llama-server, streams a completion over its SSE /completion, and exits —
       WITHOUT loading any model in-process (the server owns the only copy).
       Proves the Pi-style pivot + gets MTP spec-decode for free. Default-off. */
    if (getenv("BASI_SERVER_SELFTEST") && model_path) {
        srvgen_selftest(model_path, n_gpu_layers, ctx_override > 0 ? ctx_override : CONTEXT_SIZE);
        return 0;
    }

    /* Item 6 phase (a): BASI_SRV_CHAT_SELFTEST=1 spawns a server and drives the
       /v1/chat/completions client (messages+tools → structured tool_calls +
       separated reasoning + usage), WITHOUT loading any model in-process. Proves
       the pure-HTTP path that will let BASI drop the libllama link. */
    if (getenv("BASI_SRV_CHAT_SELFTEST") && model_path) {
        srvchat_selftest(model_path, n_gpu_layers, ctx_override > 0 ? ctx_override : CONTEXT_SIZE);
        return 0;
    }

    /* Item 6 phase (b) round-trip: serialize a mock conversation that already
       contains a tool_call + tool_result via basi_messages_to_json (OpenAI format,
       synthetic call ids), send it through the chat client, and check the model
       ANSWERS from the tool result — proving the message serialization + id
       pairing template correctly server-side. */
    if (getenv("BASI_SRV_CHAT_MSGTEST") && model_path) {
        const char *sbin = getenv("BASI_SERVER_BIN");
        if (!sbin || !*sbin) sbin = "/home/alberto/llama.cpp/build_vulkan/bin/llama-server";
        int n; const BasiToolDef *defs = basi_tool_defs(&n); basi_set_tools(defs, n);
        pid_t pid = srvgen_spawn(sbin, model_path, n_gpu_layers, 4096,
                                 "--jinja --reasoning-format auto", 8181, "/tmp/basi_srvgen.log", 300);
        if (pid < 0) { fprintf(stderr, "[msgtest] server spawn FAILED\n"); return 1; }
        struct llama_chat_message m[4] = {
            { "system",      "You are a helpful assistant. Answer concisely." },
            { "user",        "What files are in the current directory? Use bash." },
            { "tool_call",   "{\"name\":\"bash\",\"arguments\":{\"command\":\"ls\"}}" },
            { "tool_result", "{\"name\":\"bash\",\"content\":\"README.md  main.c  secret_unicorn_9f3.txt\"}" },
        };
        char *mj = basi_messages_to_json(m, 4);
        char *tj = basi_tools_to_json();
        fprintf(stderr, "[msgtest] messages JSON:\n%s\n[msgtest] tools JSON: %.80s...\n", mj, tj);
        SrvSampling samp = { .temperature = 0.0, .repeat_penalty = 1.1, .repeat_last_n = 256,
                             .min_p = 0.05, .top_k = 0, .top_p = 1.0, .seed = -1 };
        SrvChatResult *r = srvchat_complete(8181, mj, tj, &samp, 120, NULL, NULL, NULL);
        if (r) {
            fprintf(stderr, "[msgtest] finish=%s prompt_tokens=%d tool_calls=%d\n[msgtest] ANSWER: %s\n",
                    r->finish_reason ? r->finish_reason : "?", r->prompt_tokens, r->n_tool_calls,
                    r->content ? r->content : "(none)");
            srvchat_free(r);
        } else fprintf(stderr, "[msgtest] request FAILED\n");
        free(mj); free(tj);
        srvgen_kill(pid);
        return 0;
    }

    /* In --no-tools mode stdout must carry only the completion, so load chatter
     * goes to stderr. */
    fprintf(no_tools ? stderr : stdout, "BASI-CLI - Loading model...\n");
    fflush(no_tools ? stderr : stdout);

    model_init();

    /* Load model. In server-backed mode (BASI_SERVER) the weights live in the
       spawned llama-server, so BASI loads only the vocab/metadata (cheap) — used
       for its own templating, tokenization and tool-grammar derivation. */
    const bool use_server = getenv("BASI_SERVER") != NULL;
    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    model_params.vocab_only   = use_server;

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

    /* In one-shot deep-research the main context is never used for generation
       (we run deep_search in its own large context, then exit), so shrink it to
       free VRAM for the research context — important on a single GPU where two
       full-size contexts won't both fit. */
    if (oneshot_deepsearch_q) ctx_override = 4096;

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

    /* Context allocation (KV cache + compute buffers) is where VRAM actually
     * runs out — the picker's estimate is only a guess. The model weights are
     * already resident, so on failure we halve n_ctx and retry WITHOUT
     * reloading. This is the hard backstop that guarantees we never crash on a
     * mis-estimate: it degrades context length until the KV cache fits. */
    struct llama_context *ctx = NULL;
    {
        const uint32_t requested_ctx = ctx_params.n_ctx;
        const uint32_t min_ctx = 2048;
        while (1) {
            ctx = llama_init_from_model(model, ctx_params);
            if (ctx) break;
            if (ctx_params.n_ctx <= min_ctx) break;
            uint32_t reduced = ctx_params.n_ctx / 2;
            if (reduced < min_ctx) reduced = min_ctx;
            fprintf(stderr,
                "\033[33m[VRAM: context of %u didn't fit — retrying at %u]\033[0m\n",
                ctx_params.n_ctx, reduced);
            ctx_params.n_ctx   = reduced;
            ctx_params.n_batch = reduced;
        }
        if (ctx && ctx_params.n_ctx < requested_ctx) {
            fprintf(stderr,
                "\033[33m[VRAM: loaded with context %u instead of %u to fit available memory]\033[0m\n",
                ctx_params.n_ctx, requested_ctx);
        }
    }
    if (!ctx && !use_server) {
        fprintf(stderr, "Error: Failed to create context (out of VRAM even at minimum size)\n");
        llama_model_free(model);
        return 1;
    }

    /* Server-backed mode: spawn the llama-server that holds the weights + does
       generation, and route generate() to it. BASI keeps only the vocab_only
       model handle (for templating/grammar/tokenization). */
    if (use_server) {
        const char *sbin = getenv("BASI_SERVER_BIN");
        if (!sbin || !*sbin) sbin = "/home/alberto/llama.cpp/build_vulkan/bin/llama-server";
        int spec_nmax = 1;
        { const char *e = getenv("BASI_SPEC_NMAX"); if (e && *e) spec_nmax = atoi(e); }
        int srv_ctx = ctx_override > 0 ? ctx_override : CONTEXT_SIZE;

        /* Spec-decode selection, in precedence order: explicit BASI_SPEC env wins;
           else the picker's choice; else auto-enable draft-mtp for an MTP model
           (its head is exactly for this). Flash-attn follows the picker, else spec. */
        const char *spec_env = getenv("BASI_SPEC");
        int model_is_mtp = (strstr(model_path, "MTP") || strstr(model_path, "mtp")) ? 1 : 0;
        const char *spec_type = NULL;
        if (spec_env && *spec_env)      spec_type = spec_env;
        else if (picker_spec == 1)      spec_type = "draft-mtp";
        else if (picker_spec < 0 && model_is_mtp) spec_type = "draft-mtp";
        int fa_on = (picker_fa >= 0) ? picker_fa : (spec_type != NULL);

        /* "How to run llama-server for this model" IS the config now, so BASI keeps
           it as a standalone, editable script (.basi/run-llama-server.sh) and execs
           it. Reuse the user's script when it targets THIS model (respecting edits);
           regenerate when it's missing or for a different model (e.g. after /model). */
        const char *script = ".basi/run-llama-server.sh";
        SrvLaunch L = {
            .server_bin = sbin, .model_path = model_path, .ngl = n_gpu_layers,
            .ctx = srv_ctx, .host = "127.0.0.1", .port = 8181,
            .spec_type = spec_type, .spec_nmax = spec_nmax,
            .flash_attn = fa_on, .jinja = 1, .reasoning_format = "auto",
        };
        if (srvgen_script_matches(script, model_path)) {
            fprintf(stderr, "\033[90m[server mode] using launch script %s (edit it to change flags)\033[0m\n", script);
        } else {
            if (srvgen_write_launch_script(&L, script) == 0)
                fprintf(stderr, "\033[90m[server mode] wrote launch script %s\033[0m\n", script);
        }
        fprintf(stderr, "\033[90m[server mode] spawning llama-server (holds the weights)…\033[0m\n");
        g_srv_pid = srvgen_spawn_script(script, 8181, "/tmp/basi_srvgen.log", 300);
        if (g_srv_pid < 0) {
            fprintf(stderr, "Error: llama-server failed to start (see /tmp/basi_srvgen.log and %s)\n", script);
            llama_model_free(model);
            return 1;
        }
        atexit(kill_srv);
        basi_srv_port  = 8181;
        basi_srv_model = model;
        fprintf(stderr, "\033[90m[server mode] ready — generation delegated to llama-server\033[0m\n");
    }

    /* Hidden spec-decode self-test (BASI_SPEC_SELFTEST=1): plain-greedy vs
       spec-greedy A/B on a fixed prompt, report lossless+speed, then exit.
       Milestone-1 verification of the spec shim before it's wired into generate(). */
    if (getenv("BASI_SPEC_SELFTEST")) {
        basi_spec_selftest(model, ctx, model_path, (int)ctx_params.n_ctx, n_gpu_layers);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    /* Create sampler chain */
    struct llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    /* Repetition penalty — damps the degenerate repeat-loop collapse seen in long
       agentic sessions (a small model emits a malformed token, then spirals into
       "XXXX…" until it overruns the context). Added FIRST so it shapes the logits
       before min_p/temp. Mild by default (1.1 over the last 256 tokens, no
       freq/presence component); tune with BASI_REPEAT_PENALTY (1.0 disables). */
    float repeat_pen = 1.1f;
    {
        const char *rp = getenv("BASI_REPEAT_PENALTY");
        if (rp) { float v = (float)atof(rp); if (v >= 1.0f && v <= 2.0f) repeat_pen = v; }
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(256, repeat_pen, 0.0f, 0.0f));
    /* Server-backed mode decodes on llama-server, not this chain — mirror the same
       knobs into the /completion request so generation quality matches native.
       (min_p 0.05 matches SAMPLER_TAIL; seed omitted when random.) */
    if (use_server) {
        basi_srv_sampling.temperature   = temp_override >= 0 ? temp_override : 0.4;
        basi_srv_sampling.repeat_penalty = repeat_pen;
        basi_srv_sampling.repeat_last_n  = 256;
        basi_srv_sampling.min_p          = 0.05;
        basi_srv_sampling.top_k          = top_k;
        basi_srv_sampling.top_p          = top_p;
        basi_srv_sampling.seed           = (cli_seed == LLAMA_DEFAULT_SEED) ? -1 : (long) cli_seed;
    }
    /* The rest of the chain (min_p → temp → dist) is appended by SAMPLER_TAIL
       below, AFTER the tool-call grammar (when native tools are active) so the
       grammar masks invalid tokens before min_p/temp narrow the set. */
    #define SAMPLER_TAIL() do { \
        if (top_k > 0)    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k)); \
        if (top_p < 1.0f) llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1)); \
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1)); \
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp_override >= 0 ? temp_override : 0.4f)); \
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(cli_seed)); \
    } while (0)

    if (!no_tools && !oneshot_prompt) {
        print_startup_banner(model_path, (int)llama_n_ctx(ctx), n_gpu_layers);
        printf("\033[38;5;245mType your message, or /help. Empty line to quit.\033[0m\n\n");
        fflush(stdout);
    }

    /* Chat messages (dynamic array) */
    struct llama_chat_message *messages = NULL;
    size_t msg_count = 0;
    size_t msg_cap   = 0;

    /* Running token totals for /cost */
    size_t session_prompt_tokens = 0;
    size_t session_gen_tokens    = 0;

    /* Session file (set after picker; system messages are not persisted) */
    FILE *session_fp = NULL;

    /* Helper to add a message (thin alias over repl_add_message). */
    #define ADD_MESSAGE(role_str, content_str) \
        repl_add_message(&messages, &msg_count, &msg_cap, session_fp, (role_str), (content_str))

    /* Non-interactive flat completion (`-p "..." --no-tools`): one generation,
     * no tool loop, none of the tool-heavy system prompt. Built for clean,
     * scriptable output — e.g. a local teacher for distillation data-gen, where
     * the agentic loop's latency, web injection, and <tool> chatter are noise.
     * stdout carries only the completion; everything else went to stderr above. */
    if (no_tools) {
        /* Default to a minimal assistant prompt; -s overrides; -s "" drops the
         * system message entirely for a pure completion. */
        const char *sys = system_override
            ? system_override
            : "You are a helpful assistant. Answer directly and concisely.";
        if (sys && *sys) ADD_MESSAGE("system", sys);
        ADD_MESSAGE("user", oneshot_prompt);

        char *buf = malloc(FORMATTED_BUF_SZ);
        int len = apply_template(model, messages, msg_count, true,
                                 buf, FORMATTED_BUF_SZ);
        if (len < 0) {
            fprintf(stderr, "Error: Failed to apply chat template\n");
            free(buf);
            goto cleanup;
        }

        SAMPLER_TAIL();       /* --no-tools: no grammar, just finish the chain */
        generate_quiet = 1;  /* suppress streaming/decoration; we print result.text */
        GenerateResult r = generate(ctx, vocab, smpl, buf, (size_t)len);
        printf("%s\n", r.text ? r.text : "");
        fflush(stdout);
        free(r.text);
        free(buf);
        goto cleanup;
    }

    /* Native tool-calling (phase 2a): register the tool set, then ask whether
       THIS model's template supports tool calls. If so, the model emits its
       own trained <tool_call> JSON and we parse it; otherwise we fall back to
       the legacy <tool>-prose path. The system prompt is slimmed to match. */
    int tool_n = 0;
    const BasiToolDef *tool_defs = basi_tool_defs(&tool_n);
    basi_set_tools(tool_defs, tool_n);
    int native_tools = basi_tools_active(model);
    generate_native_tools = native_tools;   /* hide raw tool-call markup from the live stream */
    /* Render the answer stream as markdown, but only for the interactive REPL on
       a real terminal — -p/one-shot and piped output stay raw and parseable.
       Disable with BASI_MARKDOWN=0. */
    {
        const char *mdenv = getenv("BASI_MARKDOWN");
        generate_markdown = !oneshot_prompt && !oneshot_deepsearch_q &&
                            isatty(STDOUT_FILENO) &&
                            !(mdenv && strcmp(mdenv, "0") == 0);
    }
    printf("\033[90m[Tool mode: %s]\033[0m\n", native_tools ? "native (function-calling)" : "legacy (<tool> tags)");
    if (permission_mode == PERM_BYPASS)
        printf("\033[33m[Permissions: bypass — all tool actions auto-approved, no prompts]\033[0m\n");
    printf("\033[90m[Compaction: %s]\033[0m\n", compact_mode_name(compact_mode));
    fflush(stdout);
    if (!native_tools) basi_set_tools(NULL, 0);   /* don't advertise tools the model can't format */

    /* Phase 2b: constrain decoding to valid tool-call JSON when the model speaks
       native function-calling. Stops a small model drifting off-format after a
       few rounds (emitting malformed <tool_call> the PEG parser rejects, which
       ends the agentic loop). The grammar is LAZY — free text and thinking are
       unaffected; it only forces valid JSON once a tool call begins. Inserted
       into the chain BEFORE the min_p/temp/dist tail so it masks first. */
    if (native_tools) {
        g_tool_grammar = basi_tool_grammar_sampler(model);
        if (g_tool_grammar) {
            llama_sampler_chain_add(smpl, g_tool_grammar);
            printf("\033[90m[Tool grammar: constrained decoding active]\033[0m\n");
        } else {
            printf("\033[90m[Tool grammar: none for this format — unconstrained]\033[0m\n");
        }
        fflush(stdout);
    }
    SAMPLER_TAIL();

    /* The date is injected per-turn (see the REPL loop) so it stays fresh on
       long-lived / resumed sessions, not stamped once here. */
    char system_prompt[16384];
    build_system_prompt(system_prompt, sizeof(system_prompt), native_tools, model_path);
    ADD_MESSAGE("system", system_prompt);

    /* Non-interactive deep research: run it, print the answer, and exit —
       skipping the session picker and the REPL entirely. */
    if (oneshot_deepsearch_q) {
        char *ans = execute_deep_search(model, vocab, oneshot_deepsearch_q);
        printf("\n%s\n", ans ? ans : "(no answer)");
        fflush(stdout);
        free(ans);
        goto cleanup;
    }

    /* Session selection (interactive only). --resume <file> (used by /model to
       carry the conversation across a model switch) reloads that file and
       skips the picker; otherwise list previous sessions or start new.
       session_path is retained so /model can re-exec with --resume. */
    char *session_path = NULL;
    if (!oneshot) {
        if (resume_path) {
            session_load_into(resume_path, &messages, &msg_count, &msg_cap,
                              (int)ctx_params.n_ctx);
            session_fp = fopen(resume_path, "a");
            session_path = strdup(resume_path);
            printf("\033[90m[Resumed session: %s]\033[0m\n\n", resume_path);
            fflush(stdout);
        } else {
            char *sess_dir = session_dir_path();
            if (sess_dir) {
                char *load_path = session_picker(sess_dir);
                if (load_path) {
                    session_load_into(load_path, &messages, &msg_count, &msg_cap,
                                      (int)ctx_params.n_ctx);
                    session_fp = fopen(load_path, "a");
                    session_path = strdup(load_path);
                    printf("\033[90m[Session: %s]\033[0m\n\n", load_path);
                    free(load_path);
                } else {
                    char *new_path = NULL;
                    session_fp = session_open_new(sess_dir, &new_path);
                    if (new_path) {
                        printf("\033[90m[New session: %s]\033[0m\n\n", new_path);
                        session_path = strdup(new_path);
                        free(new_path);
                    }
                }
                fflush(stdout);
                free(sess_dir);
            }
        }
    }

    char formatted_buf[FORMATTED_BUF_SZ];
    size_t prev_len = 0;

    /* Sticky status bar: interactive sessions only (no-op on a pipe / -p).
       Derive a short model tag for the bar; the ctx meter reads `ctx` live. */
    if (!oneshot_prompt) {
        char model_tag[32];
        derive_model_tag(model_path, model_tag, sizeof model_tag);
        statusbar_enable(ctx, model_tag);
    }

    /* REPL loop (or a single injected turn in -p one-shot mode) */
    bool oneshot_done = false;
    while (1) {
        char *user_input;
        if (oneshot_prompt) {
            if (oneshot_done) break;          /* one-shot: exit after one turn */
            user_input = strdup(oneshot_prompt);
            oneshot_done = true;
        } else {
            statusbar_draw();   /* freshen the bar at the prompt (decision point) */
            user_input = read_line("\033[32m> \033[0m");
            if (!user_input) break; /* EOF */
            if (user_input[0] == '\0') {
                free(user_input);
                continue;
            }
        }

        /* Slash commands intercept (no model call) */
        if (user_input[0] == '/') {
            /* /model switches the active model. Handled here (not in
               handle_slash_command) because it re-execs and needs argv/argc,
               the session path, and the live ctx for teardown. */
            if (!oneshot_prompt && strncmp(user_input, "/model", 6) == 0 &&
                (user_input[6] == '\0' || user_input[6] == ' ')) {
                const char *marg = user_input + 6;
                while (*marg == ' ') marg++;
                try_model_switch(marg, argv, argc, model_path, n_gpu_layers,
                                 session_path, ctx);
                free(user_input);
                continue;
            }
            handle_slash_command(user_input, model, vocab, ctx,
                                 &messages, &msg_count, &msg_cap, session_fp,
                                 &prev_len, session_prompt_tokens, session_gen_tokens);
            continue;
        }

        run_agentic_turn(user_input, model, vocab, ctx, smpl, native_tools,
                         formatted_buf, &messages, &msg_count, &msg_cap,
                         session_fp, &prev_len,
                         &session_prompt_tokens, &session_gen_tokens);
        free(user_input);
    }

cleanup:
    /* Cleanup */
    statusbar_disable();   /* release the reserved bottom row before we exit */
    if (session_fp) fclose(session_fp);
    lsp_shutdown();
    embed_shutdown();
    mem_clear();
    for (size_t i = 0; i < msg_count; i++)
        free((void *)messages[i].content);
    free(messages);
    history_free_all();

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    /* --no-tools keeps stdout to the completion alone; no sign-off banner. */
    if (!no_tools) printf("\nGoodbye!\n");
    return 0;
}
