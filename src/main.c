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
#include "deepsearch.h"

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


/* ── Signal handling ───────────────────────────────────────────────── */

volatile sig_atomic_t generation_interrupted = 0;
volatile sig_atomic_t show_thinking = 0;
volatile sig_atomic_t generate_quiet = 0;

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


bool debug_mode = false;
bool bash_always_allowed = false;
bool apply_patch_always_allowed = false;
bool scaffold_always_allowed = false;

PermissionMode permission_mode = PERM_DEFAULT;

PlanPhase plan_phase = PHASE_NONE;
char *current_plan_slug = NULL;
int spike_cycles = 0;
int spike_calls  = 0;

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
            return "[DRAFTING phase — research the task using docs_*, code_context, read/grep/wc, web_search, web_fetch, readfile. BEFORE you call plan_write, call the assumptions tool with a list (one '- <item>' per line) of every unverified thing the plan depends on; if 3 or more, you'll be auto-routed to a spike phase to investigate first. Then save the Proposal-A3 plan to .basi/plans/<slug>.md via plan_write. bash/apply_patch/scaffold are BLOCKED (scaffold list is allowed).]";
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

static char *execute_tool(const char *command) {
    /* trim whitespace */
    while (*command == ' ' || *command == '\t' || *command == '\n') command++;
    if (!*command) return strdup("Error: Empty command");

    /* Phase gate (Decision #5): single check covers bash/apply_patch/scaffold
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
            "Error: Command '%s' not allowed. Use: read, head, tail, grep, wc, bash, apply_patch, scaffold, code_context, web_search, web_fetch, readfile", cmd);
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



/* ── Main ──────────────────────────────────────────────────────────── */


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

    const char *model_path = NULL;
    int n_gpu_layers = 99;
    const char *oneshot_deepsearch_q = NULL;  /* --deepsearch: run deep research and exit */
    const char *oneshot_prompt = NULL;        /* -p/--prompt: run one agent turn and exit */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            n_gpu_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--debug") == 0) {
            debug_mode = true;
        } else if ((strcmp(argv[i], "--deepsearch") == 0 || strcmp(argv[i], "-ds") == 0)
                   && i + 1 < argc) {
            oneshot_deepsearch_q = argv[++i];
        } else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0
                    || strcmp(argv[i], "--print") == 0) && i + 1 < argc) {
            oneshot_prompt = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("BASI-CLI - AI Chat Interface\n\n"
                   "Usage:\n"
                   "  basi-cli -m <model.gguf> [-ngl <n>] [-d]          interactive session\n"
                   "  basi-cli -m <model.gguf> -p \"<prompt>\"             one-shot agent turn, prints + exits\n"
                   "  basi-cli -m <model.gguf> --deepsearch \"<question>\" one-shot deep research, prints + exits\n\n"
                   "  -m              Path to GGUF model file\n"
                   "  -ngl            Number of GPU layers (default: 99)\n"
                   "  -p, --prompt    Run a single prompt non-interactively (with tools), then exit\n"
                   "  --deepsearch    Run multi-round deep research (web + KB) non-interactively, then exit\n"
                   "  -d              Debug mode (verbose tool output)\n"
                   "  -h              Show this help\n\n"
                   "Environment:\n"
                   "  BASI_MODEL             Default model path if -m not specified\n"
                   "  BASI_DEEPSEARCH_ROUNDS Max deep-research rounds (default 5)\n"
                   "  BASI_DEEPSEARCH_CTX    Deep-research context size (default 32768; lower for\n"
                   "                         interactive /deepsearch on a single GPU)\n\n");
            return 0;
        }
    }
    bool oneshot = (oneshot_deepsearch_q != NULL) || (oneshot_prompt != NULL);

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
        free(cfg.model_path);
        model_path = picked_model;
        n_gpu_layers = cfg.gpu_layers;
        ctx_override = cfg.ctx_size;
        temp_override = cfg.temperature;
    }

    /* Warm up the local SearXNG (web_search backend) while the model loads. */
    web_ensure_searxng();

    printf("BASI-CLI - Loading model...\n");
    fflush(stdout);

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

    /* The date is injected per-turn (see the REPL loop) so it stays fresh on
       long-lived / resumed sessions, not stamped once here. */
    char system_prompt[16384];
    snprintf(system_prompt, sizeof(system_prompt), "%s", SYSTEM_PROMPT_FMT);
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
                    "  apply_patch, scaffold, web_search, web_fetch, readfile, code_context,\n"
                    "  docs_toc, docs_get, docs_search, docs_recent_notes,\n"
                    "  docs_vector_search,\n"
                    "  plan_write (drafting/premortem), assumptions (drafting),\n"
                    "  spike_write (spike), plan_verify (active).\n\n");
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
            if (strncmp(user_input, "/note", 5) == 0 &&
                (user_input[5] == '\0' || user_input[5] == ' ')) {
                const char *text = user_input + 5;
                while (*text == ' ') text++;
                if (!*text) {
                    printf("\033[31m[/note: missing text — usage: /note <one-line note>]\033[0m\n");
                    fflush(stdout);
                    free(user_input);
                    continue;
                }
                if (kb_ensure_dirs() != 0) {
                    printf("\033[31m[/note: cannot create .basi/knowledge/ tree (%s)]\033[0m\n",
                           strerror(errno));
                    fflush(stdout);
                    free(user_input);
                    continue;
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
                    continue;
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
                continue;
            }
            if (strncmp(user_input, "/edit", 5) == 0 &&
                (user_input[5] == '\0' || user_input[5] == ' ')) {
                const char *arg = user_input + 5;
                while (*arg == ' ') arg++;
                if (!*arg) {
                    printf("\033[31m[/edit: missing path — usage: /edit <path-under-.basi/knowledge/>]\033[0m\n");
                    fflush(stdout);
                    free(user_input);
                    continue;
                }
                if (strstr(arg, "..") != NULL || arg[0] == '/') {
                    printf("\033[31m[/edit not allowed: path must be relative under .basi/knowledge/ and may not contain '..']\033[0m\n");
                    fflush(stdout);
                    free(user_input);
                    continue;
                }
                const char *editor = getenv("EDITOR");
                if (!editor || !*editor) {
                    printf("\033[31m[/edit not allowed: $EDITOR is not set]\033[0m\n");
                    fflush(stdout);
                    free(user_input);
                    continue;
                }
                char fullpath[1280];
                snprintf(fullpath, sizeof(fullpath), "%s/%s", KB_KNOW_DIR, arg);
                struct stat est;
                if (stat(fullpath, &est) != 0) {
                    printf("\033[31m[/edit not allowed: file not found: %s]\033[0m\n", fullpath);
                    fflush(stdout);
                    free(user_input);
                    continue;
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
                continue;
            }
            if (strcmp(user_input, "/model") == 0) {
                printf("\033[90m[Model switching is not yet implemented — exit (Ctrl-D) and restart BASI to pick a different model.]\033[0m\n");
                fflush(stdout);
                free(user_input);
                continue;
            }
            if (strncmp(user_input, "/deepsearch", 11) == 0 &&
                (user_input[11] == '\0' || user_input[11] == ' ')) {
                const char *q = user_input + 11;
                while (*q == ' ') q++;
                if (!*q) {
                    printf("\033[31m[/deepsearch: needs a question. Usage: /deepsearch <your research question>]\033[0m\n");
                    fflush(stdout);
                    free(user_input);
                    continue;
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
                llama_memory_clear(llama_get_memory(ctx), true);
                prev_len = 0;
                free(q_copy);
                free(answer);
                free(user_input);
                continue;
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
                    continue;
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
                continue;
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

cleanup:
    /* Cleanup */
    if (session_fp) fclose(session_fp);
    lsp_shutdown();
    embed_shutdown();
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
