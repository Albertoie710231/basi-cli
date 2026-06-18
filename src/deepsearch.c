#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>

#include "llama.h"

#include "util.h"
#include "globals.h"
#include "model.h"
#include "web.h"
#include "kb.h"
#include "deepsearch.h"

/* ════════════════════════════════════════════════════════════════════
   deep_search — IterResearch deep-research loop.

   Ported in structure from Alibaba DeepResearch (inference/react_agent.py,
   inference/prompt.py) and the WebResearcher "Iterative Deep-Research" paradigm
   (WebAgent/WebResearcher). The key idea, and the reason it suits a small local
   model: instead of letting one context accumulate every raw tool result (which
   suffocates a 4B), each round reconstructs a lean workspace — system + question
   + the evolving REPORT + the last OBSERVATION — and the model rewrites the
   report as its carried memory. Raw page text and the model's <think> never
   accumulate. Each visited source is distilled to goal-relevant evidence by a
   low-temperature extraction pass before it enters the report.

   The whole loop runs in its OWN llama_context built from the already-loaded
   model, so it never perturbs the main conversation's KV cache. The contract
   reuses BASI's native <tool>…</tool> action convention plus <report>/<answer>.
   ════════════════════════════════════════════════════════════════════ */

#define DEEPSEARCH_CTX        8192    /* isolated context size (tokens) */
#define DEEPSEARCH_MAX_ROUNDS    5    /* default; override BASI_DEEPSEARCH_ROUNDS */
#define DEEPSEARCH_MIN_CTX    2048    /* VRAM backoff floor */
#define DEEPSEARCH_FMT_SZ  (128*1024) /* templated-prompt buffer */
#define REPORT_MAX_CHARS      6000    /* cap on the carried report */
#define OBS_MAX_CHARS         4000    /* cap on a distilled observation */
#define RAW_FEED_MAX         12000    /* raw tool output fed to the extractor */
#define EXTRACT_SKIP_BELOW     200    /* skip extraction for tiny raw results */

/* ── prompts ────────────────────────────────────────────────────────── */

static const char *DEEPSEARCH_SYSTEM =
    "You are BASI DeepResearch, a deep research agent. You answer hard questions by "
    "investigating across multiple sources over several rounds, then synthesizing a "
    "thorough, accurate, well-cited answer.\n"
    "\n"
    "You work in ROUNDS. Each round you receive the research question, your current REPORT "
    "(your evolving memory of everything found so far), and the LATEST OBSERVATION (the "
    "distilled result of your previous action). Each round you MUST output, in this order:\n"
    "1. Optional private reasoning in <think>...</think> (discarded — never put findings here).\n"
    "2. <report>...</report> — your UPDATED report. Rewrite the previous report, folding in the "
    "new observation: keep every durable fact, figure, name, date and source URL; drop noise; "
    "note what is still unknown. This is the ONLY thing carried to the next round, so it must "
    "stand on its own.\n"
    "3. EITHER one tool call OR your final answer:\n"
    "   <tool>web_search \"query\"</tool>          search the live web\n"
    "   <tool>web_fetch \"https://url\"</tool>      read the full text of one web page\n"
    "   <tool>docs_search \"keywords\"</tool>       grep the user's local knowledge base\n"
    "   <tool>docs_get \"shelf/path.md\"</tool>     read one knowledge-base document\n"
    "   <answer>...</answer>                       your final, thorough, cited answer\n"
    "\n"
    "Rules:\n"
    "- Exactly ONE action per round (one <tool> OR one <answer>), and it MUST come after </report>.\n"
    "- Start open questions with web_search; web_fetch a specific URL when you need its full "
    "content; use docs_* to consult the user's own knowledge base.\n"
    "- Cite ONLY URLs that actually appeared in your observations. Never invent or guess a link.\n"
    "- Answer as soon as you genuinely have enough — do not pad rounds.\n"
    "Current date: ";

static const char *EXTRACTOR_SYSTEM =
    "You are an extraction assistant. Given source content and a research goal, you extract only "
    "the information relevant to the goal. You are faithful: you never invent facts or URLs, and "
    "you quote specifics (numbers, dates, names, versions, and any source URLs) exactly as they "
    "appear. If nothing in the content is relevant to the goal, you say so plainly.";

/* ── small helpers ──────────────────────────────────────────────────── */

/* Content between <tag> and </tag>, malloc'd. If the close tag is missing
   (e.g. generation was cut off), takes everything to the end. NULL if no open
   tag. */
static char *extract_tagged(const char *text, const char *tag) {
    char open[32], close[32];
    snprintf(open, sizeof(open), "<%s>", tag);
    snprintf(close, sizeof(close), "</%s>", tag);
    const char *s = strstr(text, open);
    if (!s) return NULL;
    s += strlen(open);
    const char *e = strstr(s, close);
    size_t n = e ? (size_t)(e - s) : strlen(s);
    char *out = malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

/* strdup, but truncate to max chars with a marker if longer. */
static char *cap_dup(const char *s, size_t max) {
    if (!s) return strdup("");
    size_t n = strlen(s);
    if (n <= max) return strdup(s);
    char *o = malloc(max + 24);
    if (!o) return strdup("");
    memcpy(o, s, max);
    strcpy(o + max, "\n…[truncated]");   /* 15 bytes + NUL; fits in +24 */
    return o;
}

/* Strip leading/trailing ASCII whitespace in place (returns s). */
static char *trim(char *s) {
    if (!s) return s;
    char *p = s;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (p != s) memmove(s, p, strlen(p) + 1);
    size_t n = strlen(s);
    while (n > 0 && (s[n-1] == ' ' || s[n-1] == '\t' || s[n-1] == '\n' || s[n-1] == '\r'))
        s[--n] = '\0';
    return s;
}

/* Run one single-shot, quiet generation of a (system,user) exchange in `dctx`.
   Clears the KV first so generate()'s is_first logic restarts cleanly. Returns
   the malloc'd model output (caller frees), or NULL on template failure. */
static char *ds_generate(struct llama_context *dctx, const struct llama_vocab *vocab,
                         struct llama_sampler *smpl, const char *tmpl,
                         const char *system, const char *user, char *fmt_buf) {
    struct llama_chat_message msgs[2];
    msgs[0].role = "system"; msgs[0].content = system;
    msgs[1].role = "user";   msgs[1].content = user;
    int len = apply_template(tmpl, msgs, 2, true, fmt_buf, DEEPSEARCH_FMT_SZ);
    if (len < 0) return NULL;
    llama_memory_clear(llama_get_memory(dctx), true);
    GenerateResult r = generate(dctx, vocab, smpl, fmt_buf, (size_t)len);
    return r.text;
}

/* Dispatch one research action to the existing tool entry points. Returns the
   malloc'd raw result; sets *phase to a short progress label. */
static char *dispatch_research_tool(const char *name, const char *arg, const char **phase) {
    if (strcasecmp(name, "web_search") == 0) { *phase = "searching web"; return execute_web_search(arg, NULL); }
    if (strcasecmp(name, "web_fetch") == 0)  { *phase = "reading page";  return execute_web_fetch(arg); }
    if (strcasecmp(name, "docs_search") == 0){ *phase = "searching KB";  return execute_docs_search(arg); }
    if (strcasecmp(name, "docs_get") == 0)   { *phase = "reading KB doc";return execute_docs_get(arg); }
    *phase = "unknown action";
    char *msg = malloc(192);
    if (msg) snprintf(msg, 192,
        "Error: unknown tool '%.32s'. Use one of: web_search, web_fetch, docs_search, docs_get.", name);
    return msg;
}

/* ── interrupt handling (self-contained; restored on exit) ──────────── */

static void ds_sigint(int sig) { (void)sig; generation_interrupted = 1; }

/* ── main entry ─────────────────────────────────────────────────────── */

char *execute_deep_search(struct llama_model *model,
                          const struct llama_vocab *vocab,
                          const char *question) {
    if (!model || !vocab || !question || !question[0])
        return strdup("Error: deep_search requires a question.");

    int max_rounds = DEEPSEARCH_MAX_ROUNDS;
    const char *env = getenv("BASI_DEEPSEARCH_ROUNDS");
    if (env && env[0]) {
        int v = atoi(env);
        if (v >= 1 && v <= 50) max_rounds = v;
    }

    /* Isolated context over the same model, with the same VRAM halving-retry
       backoff the main context uses (main.c) so tight-VRAM boxes degrade. */
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = DEEPSEARCH_CTX;
    cp.n_batch = cp.n_ctx;
    int n_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (n_cores > 2) { cp.n_threads = n_cores / 2; cp.n_threads_batch = n_cores / 2; }

    struct llama_context *dctx = NULL;
    while (1) {
        dctx = llama_init_from_model(model, cp);
        if (dctx) break;
        if (cp.n_ctx <= DEEPSEARCH_MIN_CTX) break;
        uint32_t reduced = cp.n_ctx / 2;
        if (reduced < DEEPSEARCH_MIN_CTX) reduced = DEEPSEARCH_MIN_CTX;
        cp.n_ctx = reduced; cp.n_batch = reduced;
    }
    if (!dctx)
        return strdup("Error: deep_search could not allocate a research context (out of VRAM).");

    /* Two sampler chains: reasoning/synthesis (mild temp) and faithful, near-
       deterministic extraction. Not the main chain — generate() mutates sampler
       state and we must not perturb the chat session's sampler. */
    struct llama_sampler *dsmpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(dsmpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(dsmpl, llama_sampler_init_temp(0.4f));
    llama_sampler_chain_add(dsmpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    struct llama_sampler *xsmpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(xsmpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(xsmpl, llama_sampler_init_temp(0.1f));
    llama_sampler_chain_add(xsmpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    char *fmt_buf = malloc(DEEPSEARCH_FMT_SZ);
    if (!fmt_buf) {
        llama_sampler_free(xsmpl); llama_sampler_free(dsmpl); llama_free(dctx);
        return strdup("Error: out of memory.");
    }

    const char *tmpl = llama_model_chat_template(model, NULL);

    /* System prompt with today's date appended (matches the chat session's
       trust-the-date convention). */
    char system_prompt[4096];
    {
        time_t t = time(NULL);
        struct tm tmv;
        char datebuf[32] = "";
        if (localtime_r(&t, &tmv)) strftime(datebuf, sizeof(datebuf), "%Y-%m-%d", &tmv);
        snprintf(system_prompt, sizeof(system_prompt), "%s%s", DEEPSEARCH_SYSTEM, datebuf);
    }

    /* Quiet the model's internal generations; we print our own progress. */
    sig_atomic_t prev_quiet = generate_quiet;
    generate_quiet = 1;

    /* Catch Ctrl+C so a long run can be stopped without killing the process. */
    struct sigaction old_sa, sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = ds_sigint;
    sa.sa_flags = SA_RESTART;
    sigaction(SIGINT, &sa, &old_sa);
    generation_interrupted = 0;

    printf("\033[36m[deepsearch] researching (up to %d rounds; Ctrl+C to stop early): %.120s\033[0m\n",
           max_rounds, question);
    fflush(stdout);

    char *report  = strdup("");
    char *last_obs = strdup("");
    char *answer  = NULL;
    int round = 0;

    for (round = 1; round <= max_rounds; round++) {
        /* Build the lean round workspace. */
        StringBuf u; sb_init(&u);
        sb_append_str(&u, "## Research question\n");
        sb_append_str(&u, question);
        sb_append_str(&u, "\n\n## Current report\n");
        sb_append_str(&u, report[0] ? report : "(empty — this is the first round; start by investigating)");
        sb_append_str(&u, "\n\n## Latest observation\n");
        sb_append_str(&u, last_obs[0] ? last_obs : "(none yet)");
        sb_append_str(&u, "\n\nNow output your updated <report>...</report> followed by exactly one "
                          "action (<tool>...</tool> or <answer>...</answer>).");
        char *user = sb_to_str(&u);

        char *out = ds_generate(dctx, vocab, dsmpl, tmpl, system_prompt, user, fmt_buf);
        free(user);
        if (!out) { printf("\033[31m[deepsearch] template error; stopping.\033[0m\n"); fflush(stdout); break; }

        if (generation_interrupted) { free(out); printf("\033[33m[deepsearch] interrupted.\033[0m\n"); fflush(stdout); break; }

        /* Carry forward the report (keep previous if the model omitted it). */
        char *new_report = extract_tagged(out, "report");
        if (new_report) {
            trim(new_report);
            if (new_report[0]) { free(report); report = cap_dup(new_report, REPORT_MAX_CHARS); }
            free(new_report);
        }

        /* Terminal: an answer was produced. */
        char *ans = extract_tagged(out, "answer");
        if (ans) {
            trim(ans);
            answer = ans;
            free(out);
            printf("\033[36m[deepsearch %d/%d] answer ready\033[0m\n", round, max_rounds);
            fflush(stdout);
            break;
        }

        /* Otherwise: one tool action. */
        size_t tlen = 0;
        const char *tbody = extract_tool_call(out, &tlen);
        if (!tbody) {
            free(out);
            free(last_obs);
            last_obs = strdup("Your previous response contained neither a <tool> action nor an "
                              "<answer>. Emit exactly one <tool>...</tool> action, or <answer>...</answer> "
                              "if you have enough to conclude.");
            printf("\033[33m[deepsearch %d/%d] no action — reprompting\033[0m\n", round, max_rounds);
            fflush(stdout);
            continue;
        }
        char *cmd = malloc(tlen + 1);
        memcpy(cmd, tbody, tlen); cmd[tlen] = '\0';
        free(out);

        ArgList al = tokenize_command(cmd);
        free(cmd);
        if (al.count < 1) {
            arglist_free(&al);
            free(last_obs);
            last_obs = strdup("Your tool call was empty. Use e.g. <tool>web_search \"your query\"</tool>.");
            continue;
        }
        const char *tool_name = al.args[0];
        const char *tool_arg  = al.count >= 2 ? al.args[1] : "";

        const char *phase = "";
        char *raw = dispatch_research_tool(tool_name, tool_arg, &phase);
        printf("\033[90m[deepsearch %d/%d] %s: %.80s\033[0m\n", round, max_rounds, phase, tool_arg);
        fflush(stdout);
        arglist_free(&al);
        if (!raw) raw = strdup("(no result)");

        if (generation_interrupted) { free(raw); printf("\033[33m[deepsearch] interrupted.\033[0m\n"); fflush(stdout); break; }

        /* Goal-directed distillation of the raw result into the next
           observation. Skip the extra model call for tiny results. */
        if (strlen(raw) < EXTRACT_SKIP_BELOW) {
            free(last_obs);
            last_obs = cap_dup(raw, OBS_MAX_CHARS);
        } else {
            char *capped = cap_dup(raw, RAW_FEED_MAX);
            StringBuf x; sb_init(&x);
            sb_append_str(&x, "## Goal\n");
            sb_append_str(&x, question);
            sb_append_str(&x, "\n\n## Source content\n");
            sb_append_str(&x, capped);
            sb_append_str(&x, "\n\nExtract the evidence in this content relevant to the goal: quote the "
                              "specific facts, figures and any source URLs, then add a 2-3 sentence summary. "
                              "Be faithful and compact; if nothing is relevant, say so.");
            char *xuser = sb_to_str(&x);
            free(capped);
            char *distilled = ds_generate(dctx, vocab, xsmpl, tmpl, EXTRACTOR_SYSTEM, xuser, fmt_buf);
            free(xuser);
            free(last_obs);
            if (distilled && distilled[0]) {
                trim(distilled);
                last_obs = cap_dup(distilled, OBS_MAX_CHARS);
            } else {
                last_obs = cap_dup(raw, OBS_MAX_CHARS);   /* fall back to raw if extraction failed */
            }
            free(distilled);
        }
        free(raw);
    }

    /* No explicit answer (budget hit or interrupted): synthesize from the report. */
    if (!answer) {
        if (!generation_interrupted)
            printf("\033[36m[deepsearch] round budget reached — synthesizing final answer\033[0m\n");
        fflush(stdout);
        StringBuf f; sb_init(&f);
        sb_append_str(&f, "## Research question\n");
        sb_append_str(&f, question);
        sb_append_str(&f, "\n\n## Your report\n");
        sb_append_str(&f, report[0] ? report : "(you gathered little; answer as best you can)");
        sb_append_str(&f, "\n\nBased ONLY on the report above, give your final answer now inside "
                          "<answer>...</answer>. Be thorough and cite the source URLs your report recorded.");
        char *fuser = sb_to_str(&f);
        char *out = ds_generate(dctx, vocab, dsmpl, tmpl, system_prompt, fuser, fmt_buf);
        free(fuser);
        if (out) {
            char *ans = extract_tagged(out, "answer");
            answer = ans ? (trim(ans), ans) : strdup(out);
            free(out);
        }
    }

    if (!answer || !answer[0]) {
        free(answer);
        StringBuf e; sb_init(&e);
        sb_append_str(&e, "deep_search produced no final answer. Best report so far:\n\n");
        sb_append_str(&e, report[0] ? report : "(nothing gathered)");
        answer = sb_to_str(&e);
    }

    /* Teardown — never free the model (owned by main). */
    sigaction(SIGINT, &old_sa, NULL);
    generation_interrupted = 0;
    generate_quiet = prev_quiet;
    free(report);
    free(last_obs);
    free(fmt_buf);
    llama_sampler_free(xsmpl);
    llama_sampler_free(dsmpl);
    llama_free(dctx);

    return answer;
}
