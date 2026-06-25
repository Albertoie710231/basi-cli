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

/* MAX_TOKENS, CONTEXT_SIZE → globals.h */
#define MAX_FILE_TOKENS     2000
#define MAX_TOOL_RESULT_SZ  16000  /* max chars in a tool result (~4000 tokens) */
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
    "- bash <command> : Run an arbitrary shell command via 'bash -c'. ALWAYS requires user approval before execution; the user may deny. Use for builds (make, cargo, npm), tests, git operations, or anything not covered by the other tools. Prefer specific commands; avoid destructive operations (rm -rf, package installs) without explaining first. Output combines stdout and stderr.\n"
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
    uint32_t    cli_seed;           /* --seed: RNG seed for sampling */
    bool        bypass;             /* --yolo/--bypass: auto-approve all tool actions */
    bool        want_exit;          /* -h/--help: caller should return exit_code */
    int         exit_code;
} Cli;

static Cli parse_args(int argc, char **argv) {
    Cli c = {
        .model_path = NULL, .n_gpu_layers = 99, .ngl_set = false,
        .deepsearch_q = NULL, .prompt = NULL, .no_tools = false,
        .system_override = NULL, .cli_ctx = 0, .cli_temp = -1.0f,
        .cli_seed = LLAMA_DEFAULT_SEED, .bypass = false,
        .want_exit = false, .exit_code = 0,
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
                   "  -d              Debug mode (verbose tool output)\n"
                   "  -h              Show this help\n\n"
                   "Environment:\n"
                   "  BASI_MODEL             Default model path if -m not specified\n"
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
static int context_used_tokens(struct llama_context *ctx) {
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
                    "  /model                switch model (requires restart)\n"
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
                if (was_raw) disable_raw_mode();
                int rc = system(cmd);
                if (was_raw) enable_raw_mode();
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
                if (was_raw) disable_raw_mode();
                int rc = system(cmdbuf);
                if (was_raw) enable_raw_mode();
                if (rc != 0) printf("\033[31m[Editor exited with status %d]\033[0m\n", rc);
                else printf("\033[90m[%s edited]\033[0m\n", fullpath);
                fflush(stdout);
                free(user_input);
                return;
            }
            if (strcmp(user_input, "/model") == 0) {
                printf("\033[90m[Model switching is not yet implemented — exit (Ctrl-D) and restart BASI to pick a different model.]\033[0m\n");
                fflush(stdout);
                free(user_input);
                return;
            }
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
            char meter[80];
            format_context_meter(ctx, meter, sizeof meter);
            printf("\033[90m[ Prompt: %.1f t/s | Generation: %.1f t/s | %s ]\033[0m\n",
                   prompt_tps, gen_tps, meter);
            fflush(stdout);

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
                    printf("\033[90m[Executing: %s]\033[0m\n", nm);
                    fflush(stdout);
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
                if (cmd_str) {                 /* legacy path: build → execute */
                    printf("\033[90m[Executing: %s]\033[0m\n", cmd_str);
                    fflush(stdout);
                    tool_result = execute_tool(cmd_str);
                    free(cmd_str);
                } else if (!tool_result) {     /* native path hit an unknown tool */
                    tool_result = malloc(256);
                    snprintf(tool_result, 256,
                        "Error: unknown tool '%s' — it is not one of the available functions.", unknown_tool);
                    printf("\033[90m[Unknown tool: %s]\033[0m\n", unknown_tool ? unknown_tool : "?");
                    fflush(stdout);
                }
                free(unknown_tool);
                if (ncalls) basi_free_tool_calls(ncalls, n_ncalls);

                /* Truncate tool result if too large */
                size_t tr_len = strlen(tool_result);
                if (tr_len > MAX_TOOL_RESULT_SZ) {
                    printf("\033[90m[Truncated: %zu → %d chars]\033[0m\n",
                           tr_len, MAX_TOOL_RESULT_SZ);
                    strcpy(tool_result + MAX_TOOL_RESULT_SZ - 40,
                           "\n\n[... content truncated ...]");
                }

                llama_memory_t mem = llama_get_memory(ctx);
                int used = llama_memory_seq_pos_max(mem, 0) + 1;
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
                int prev = apply_template(
                    model, messages, msg_count - 1, false, NULL, 0);
                prompt = formatted_buf + prev;
                prompt_len = (size_t)next_len - (size_t)prev;

                printf("\n");
                fflush(stdout);
            } else {
                if (ncalls) basi_free_tool_calls(ncalls, n_ncalls);
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
            char meter[80];
            format_context_meter(ctx, meter, sizeof meter);
            printf("\033[90m[ Generation: %.1f t/s | %s ]\033[0m\n", gen_tps, meter);
            fflush(stdout);

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

    if (!model_path) {
        model_path = getenv("BASI_MODEL");
    }
    /* No model specified — show interactive picker with settings */
    static char picked_model[1024];
    int ctx_override = 0;
    float temp_override = -1;
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
    }

    /* Explicit CLI knobs win over picker / built-in defaults. */
    if (cli_ctx  > 0)   ctx_override  = cli_ctx;
    if (cli_temp >= 0)  temp_override = cli_temp;

    /* Warm up the local SearXNG (web_search backend) while the model loads.
     * --no-tools never touches the web, so don't spin SearXNG up for it. */
    if (!no_tools) web_ensure_searxng();

    /* In --no-tools mode stdout must carry only the completion, so load chatter
     * goes to stderr. */
    fprintf(no_tools ? stderr : stdout, "BASI-CLI - Loading model...\n");
    fflush(no_tools ? stderr : stdout);

    model_init();

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
    if (!ctx) {
        fprintf(stderr, "Error: Failed to create context (out of VRAM even at minimum size)\n");
        llama_model_free(model);
        return 1;
    }

    /* Create sampler chain */
    struct llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp_override >= 0 ? temp_override : 0.4f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(cli_seed));

    if (!no_tools && !oneshot_prompt) {
        printf("Model loaded. Type your message (empty line to quit).\n\n");
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
    printf("\033[90m[Tool mode: %s]\033[0m\n", native_tools ? "native (function-calling)" : "legacy (<tool> tags)");
    if (permission_mode == PERM_BYPASS)
        printf("\033[33m[Permissions: bypass — all tool actions auto-approved, no prompts]\033[0m\n");
    printf("\033[90m[Compaction: %s]\033[0m\n", compact_mode_name(compact_mode));
    fflush(stdout);
    if (!native_tools) basi_set_tools(NULL, 0);   /* don't advertise tools the model can't format */

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

    /* Session picker (interactive only): list previous sessions, or start new */
    if (!oneshot) {
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

    /* REPL loop (or a single injected turn in -p one-shot mode) */
    bool oneshot_done = false;
    while (1) {
        char *user_input;
        if (oneshot_prompt) {
            if (oneshot_done) break;          /* one-shot: exit after one turn */
            user_input = strdup(oneshot_prompt);
            oneshot_done = true;
        } else {
            user_input = read_line("\033[32m> \033[0m");
            if (!user_input) break; /* EOF */
            if (user_input[0] == '\0') {
                free(user_input);
                continue;
            }
        }

        /* Slash commands intercept (no model call) */
        if (user_input[0] == '/') {
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
