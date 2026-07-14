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
   deep_search — multi-round deep-research loop (ReAct, ported in structure
   from Alibaba DeepResearch inference/react_agent.py + inference/prompt.py).

   The agent keeps a running CHAT of its own work — system, question, then
   alternating assistant(action) / user(result) turns — and the WHOLE chat is
   fed back each round. The model therefore sees every search it has already
   run and every finding so far, so it doesn't repeat itself and can synthesize
   from the full picture. This is affordable because each page is first DISTILLED
   to goal-relevant evidence by a low-temperature extraction pass (the only
   bulky thing — raw page text — never enters the chat), so the history stays
   small and fits a 32k context many rounds deep.

   Runs in its OWN isolated llama_context built from the loaded model, so it
   never perturbs the chat session's KV cache (precedent: src/embed.c). Tools:
   web_search / web_fetch / docs_search / docs_get. Actions use BASI's native
   <tool>…</tool> convention; the final answer is <answer>…</answer>.
   ════════════════════════════════════════════════════════════════════ */

#define DEEPSEARCH_CTX       32768    /* isolated context — holds the full chat */
#define DEEPSEARCH_MAX_ROUNDS    5    /* default; override BASI_DEEPSEARCH_ROUNDS */
#define DEEPSEARCH_MIN_CTX    2048    /* VRAM backoff floor */
#define DEEPSEARCH_FMT_SZ  (512*1024) /* templated full-chat prompt buffer */
#define OBS_MAX_CHARS         4000    /* cap on a distilled result before it enters the chat */
#define RAW_FEED_MAX         12000    /* raw tool output fed to the extractor */
#define EXTRACT_SKIP_BELOW     200    /* skip extraction for tiny raw results */

/* ── prompts ────────────────────────────────────────────────────────── */

static const char *DEEPSEARCH_SYSTEM =
    "You are BASI DeepResearch, a deep research agent. You answer a hard question by "
    "investigating across multiple sources over several rounds, then giving a thorough, "
    "accurate, well-cited answer.\n"
    "\n"
    "This conversation is your research log: it already contains every search you have run "
    "and every finding so far. Read it before acting — do NOT repeat a search you have "
    "already done; build on what you have, or explore a genuinely new angle (a specific "
    "name, paper, arXiv id, or sub-topic).\n"
    "\n"
    "Each turn: optionally reason in <think>...</think> (discarded), then take EXACTLY ONE action:\n"
    "  <tool>web_search \"query\"</tool>          search the live web\n"
    "  <tool>web_fetch \"https://url\"</tool>      read the full text of one web page\n"
    "  <tool>docs_search \"keywords\"</tool>       grep the user's local knowledge base\n"
    "  <tool>docs_get \"shelf/path.md\"</tool>     read one knowledge-base document\n"
    "  <answer>...</answer>                       your final, thorough, cited answer\n"
    "\n"
    "Rules:\n"
    "- Exactly ONE action per turn (one <tool> OR one <answer>).\n"
    "- Start with web_search; web_fetch a specific URL when you need its full content; use "
    "docs_* for the user's own knowledge base.\n"
    "- Prefer specific names/papers over re-issuing broad queries you've already tried.\n"
    "- Cite ONLY URLs that actually appeared in your findings. Never invent a link.\n"
    "- Give your <answer> as soon as you genuinely have enough — do not pad rounds.\n"
    "Current date: ";

static const char *EXTRACTOR_SYSTEM =
    "You are an extraction assistant. Given source content and a research goal, you extract only "
    "the information relevant to the goal. You are faithful: you never invent facts or URLs, and "
    "you quote specifics (numbers, dates, names, versions, and any source URLs) exactly as they "
    "appear. If nothing in the content is relevant to the goal, you say so plainly.";

/* ── small helpers ──────────────────────────────────────────────────── */

/* Content between <tag> and </tag>, malloc'd. If the close tag is missing
   (generation cut off), takes everything to the end. NULL if no open tag. */
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

/* Pointer to the ACTION region of the model output — never inside reasoning:
     - <think> opened AND closed  -> text after the last </think>;
     - <think> opened, NOT closed -> "" (still thinking / cut off), so a <tool>
       or <answer> mentioned *inside* the reasoning is NOT taken as the real
       action (the round becomes a no-action reprompt instead of executing a
       hypothetical/half-written tool call from the think);
     - no <think> at all          -> the whole output (non-thinking models). */
static const char *after_think(const char *s) {
    const char *close = NULL, *p = s, *q;
    while ((q = strstr(p, "</think>")) != NULL) { close = q; p = q + 8; }
    if (close) return close + 8;
    if (strstr(s, "<think>")) return s + strlen(s);   /* unclosed think → no action */
    return s;
}

/* strdup, but truncate to max chars with a marker if longer. */
static char *cap_dup(const char *s, size_t max) {
    if (!s) return strdup("");
    size_t n = strlen(s);
    if (n <= max) return strdup(s);
    char *o = malloc(max + 24);
    if (!o) return strdup("");
    memcpy(o, s, max);
    strcpy(o + max, "\n…[truncated]");
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

/* ── growing chat message list ──────────────────────────────────────── */

/* Appends a turn, TAKING OWNERSHIP of `content` (already malloc'd). `role` is a
   string literal ("system"/"user"/"assistant"), not owned. */
static void msgs_add(struct llama_chat_message **m, size_t *n, size_t *cap,
                     const char *role, char *content) {
    if (*n >= *cap) {
        *cap = *cap ? *cap * 2 : 16;
        *m = realloc(*m, *cap * sizeof(**m));
    }
    (*m)[*n].role    = role;
    (*m)[*n].content = content;
    (*n)++;
}

/* Single-shot quiet generation over a message list. Clears the KV first so
   generate()'s is_first logic restarts cleanly, then templates + decodes the
   whole conversation. Returns malloc'd model output (caller frees), or NULL on
   template failure. */
static char *ds_generate(struct llama_context *dctx, const struct llama_vocab *vocab,
                         struct llama_sampler *smpl, const struct llama_model *model,
                         const struct llama_chat_message *msgs, size_t nmsg, char *fmt_buf) {
    int len = apply_template(model, msgs, nmsg, true, fmt_buf, DEEPSEARCH_FMT_SZ);
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

/* Distill a raw tool result into goal-relevant evidence (the only thing that
   enters the chat). Tiny results pass through verbatim. Returns malloc'd. */
static char *distill_result(struct llama_context *dctx, const struct llama_vocab *vocab,
                            struct llama_sampler *xsmpl, const struct llama_model *model,
                            const char *question, const char *raw, char *fmt_buf) {
    if (strlen(raw) < EXTRACT_SKIP_BELOW) return cap_dup(raw, OBS_MAX_CHARS);

    char *capped = cap_dup(raw, RAW_FEED_MAX);
    StringBuf x; sb_init(&x);
    sb_append_str(&x, "## Goal\n");
    sb_append_str(&x, question);
    sb_append_str(&x, "\n\n## Source content\n");
    sb_append_str(&x, capped);
    sb_append_str(&x, "\n\nExtract the evidence in this content relevant to the goal: quote the "
                      "specific facts, figures and any source URLs, then add a 2-3 sentence summary. "
                      "Be faithful and compact; if nothing is relevant, say so.");
    free(capped);

    struct llama_chat_message xm[2];
    xm[0].role = "system"; xm[0].content = (char *)EXTRACTOR_SYSTEM;
    xm[1].role = "user";   xm[1].content = x.data;
    char *distilled = ds_generate(dctx, vocab, xsmpl, model, xm, 2, fmt_buf);
    sb_free(&x);

    if (distilled && distilled[0]) { trim(distilled); char *c = cap_dup(distilled, OBS_MAX_CHARS); free(distilled); return c; }
    free(distilled);
    return cap_dup(raw, OBS_MAX_CHARS);   /* fall back to raw if extraction failed */
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

    /* Isolated context over the same model, with the main context's VRAM
       halving-retry backoff so tight-VRAM boxes degrade instead of failing.
       BASI_DEEPSEARCH_CTX overrides the size — lower it (e.g. 8192) for
       interactive /deepsearch on a single GPU, where the full-size chat context
       must coexist with the main conversation's context. */
    uint32_t want_ctx = DEEPSEARCH_CTX;
    const char *ctx_env = getenv("BASI_DEEPSEARCH_CTX");
    if (ctx_env && ctx_env[0]) {
        int v = atoi(ctx_env);
        if (v >= (int)DEEPSEARCH_MIN_CTX && v <= 131072) want_ctx = (uint32_t)v;
    }
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = want_ctx;
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

    /* Two sampler chains (not the chat session's): mild-temp reasoning and
       near-deterministic, faithful extraction. */
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

    char system_prompt[4096];
    {
        time_t t = time(NULL);
        struct tm tmv;
        char datebuf[32] = "";
        if (localtime_r(&t, &tmv)) strftime(datebuf, sizeof(datebuf), "%Y-%m-%d", &tmv);
        snprintf(system_prompt, sizeof(system_prompt), "%s%s", DEEPSEARCH_SYSTEM, datebuf);
    }

    sig_atomic_t prev_quiet = generate_quiet;
    generate_quiet = 1;
    /* Keep <think> in the model's turns so reasoning models stay consistent. */
    sig_atomic_t prev_keep_think = generate_keep_think;
    generate_keep_think = 1;
    /* Server mode: keep the MAIN tool-call grammar out of deepsearch's rounds and
       synthesis — deepsearch drives its own ReAct format, and in native mode its
       sampler carries no grammar. Without this the grammar leaks in and the final
       synthesis degenerates into a stray tool query instead of an answer. */
    int prev_suppress_grammar = basi_srv_suppress_grammar;
    basi_srv_suppress_grammar = 1;

    struct sigaction old_sa, sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = ds_sigint;
    sa.sa_flags = SA_RESTART;
    sigaction(SIGINT, &sa, &old_sa);
    generation_interrupted = 0;

    printf("\033[36m[deepsearch] researching (up to %d rounds; Ctrl+C to stop early): %.120s\033[0m\n",
           max_rounds, question);
    fflush(stdout);

    /* The research chat: system, question, then assistant(action)/user(result). */
    struct llama_chat_message *msgs = NULL;
    size_t nmsg = 0, capmsg = 0;
    msgs_add(&msgs, &nmsg, &capmsg, "system", strdup(system_prompt));
    msgs_add(&msgs, &nmsg, &capmsg, "user",   strdup(question));

    char *answer = NULL;
    int tools_run = 0;   /* must-search-before-answer guard: # of real searches/fetches done */
    int round;
    for (round = 1; round <= max_rounds; round++) {
        char *out = ds_generate(dctx, vocab, dsmpl, model, msgs, nmsg, fmt_buf);
        if (!out) { printf("\033[31m[deepsearch] template error; stopping.\033[0m\n"); fflush(stdout); break; }
        if (generation_interrupted) { free(out); printf("\033[33m[deepsearch] interrupted.\033[0m\n"); fflush(stdout); break; }

        if (getenv("BASI_DEBUG_DEEPSEARCH"))
            fprintf(stderr, "\n\033[35m[ds raw round %d]\033[0m\n%s\n\033[35m[/ds raw round %d]\033[0m\n",
                    round, out, round);

        /* The assistant turn IS the model output, INCLUDING its <think> block
           (generate_keep_think) so a reasoning model's history stays consistent.
           Transfer ownership into the chat; parse actions from AFTER the think. */
        msgs_add(&msgs, &nmsg, &capmsg, "assistant", out);
        const char *act = after_think(msgs[nmsg - 1].content);

        char *ans = extract_tagged(act, "answer");
        if (ans) {
            if (tools_run == 0) {
                /* Reject a premature answer: the model must investigate before
                   concluding. This stops over-confident (esp. reasoning) models
                   from answering from stale memory without ever searching. */
                free(ans);
                msgs_add(&msgs, &nmsg, &capmsg, "user",
                         strdup("You tried to answer WITHOUT searching. Your training data is stale and "
                                "must not be trusted for this. Investigate FIRST: issue one "
                                "<tool>web_search \"...\"</tool> (or web_fetch/docs_*). Only answer after "
                                "you have real findings."));
                printf("\033[33m[deepsearch %d/%d] answer rejected — must search first\033[0m\n", round, max_rounds);
                fflush(stdout);
                continue;
            }
            answer = trim(ans);
            printf("\033[36m[deepsearch %d/%d] answer ready\033[0m\n", round, max_rounds);
            fflush(stdout);
            break;
        }

        size_t tlen = 0;
        const char *tbody = extract_tool_call(act, &tlen);
        if (!tbody) {
            msgs_add(&msgs, &nmsg, &capmsg, "user",
                     strdup("Your last turn had neither a <tool> action nor an <answer>. "
                            "Emit exactly one <tool>...</tool>, or <answer>...</answer> if you can conclude."));
            printf("\033[33m[deepsearch %d/%d] no action — reprompting\033[0m\n", round, max_rounds);
            fflush(stdout);
            continue;
        }

        char *cmd = malloc(tlen + 1);
        memcpy(cmd, tbody, tlen); cmd[tlen] = '\0';
        ArgList al = tokenize_command(cmd);
        free(cmd);
        if (al.count < 1) {
            arglist_free(&al);
            msgs_add(&msgs, &nmsg, &capmsg, "user",
                     strdup("Your tool call was empty. Use e.g. <tool>web_search \"your query\"</tool>."));
            continue;
        }
        const char *tool_name = al.args[0];
        const char *tool_arg  = al.count >= 2 ? al.args[1] : "";

        /* Reasoning models often emit a bare "<tool>web_search</tool>" and leave
           the actual query in their prose/thinking. Reject an argument-less call
           with a precise correction instead of running an empty search. */
        if ((strcasecmp(tool_name, "web_search") == 0 || strcasecmp(tool_name, "web_fetch") == 0 ||
             strcasecmp(tool_name, "docs_search") == 0 || strcasecmp(tool_name, "docs_get") == 0)
            && !tool_arg[0]) {
            StringBuf m; sb_init(&m);
            sb_append_str(&m, "Your <tool>");
            sb_append_str(&m, tool_name);
            sb_append_str(&m, "</tool> call was missing its argument. Put the query/URL/path INSIDE the "
                              "tag, quoted, e.g. <tool>");
            sb_append_str(&m, tool_name);
            sb_append_str(&m, " \"your query here\"</tool> — do not describe it in prose. Re-issue it now.");
            printf("\033[33m[deepsearch %d/%d] %s had no argument — reprompting\033[0m\n",
                   round, max_rounds, tool_name);
            fflush(stdout);
            arglist_free(&al);
            msgs_add(&msgs, &nmsg, &capmsg, "user", sb_to_str(&m));
            continue;
        }

        const char *phase = "";
        char *raw = dispatch_research_tool(tool_name, tool_arg, &phase);
        if (strcasecmp(tool_name, "web_search") == 0 || strcasecmp(tool_name, "web_fetch") == 0 ||
            strcasecmp(tool_name, "docs_search") == 0 || strcasecmp(tool_name, "docs_get") == 0)
            tools_run++;   /* a real investigation happened — answers now allowed */
        printf("\033[90m[deepsearch %d/%d] %s: %.80s\033[0m\n", round, max_rounds, phase, tool_arg);
        fflush(stdout);
        if (!raw) raw = strdup("(no result)");

        if (generation_interrupted) { free(raw); arglist_free(&al); printf("\033[33m[deepsearch] interrupted.\033[0m\n"); fflush(stdout); break; }

        char *obs = distill_result(dctx, vocab, xsmpl, model, question, raw, fmt_buf);
        free(raw);

        /* Record the result as a user turn, prefixed with the action it answers
           so the chat clearly shows what was searched (anti-repetition). */
        StringBuf o; sb_init(&o);
        sb_append_str(&o, "Result of ");
        sb_append_str(&o, tool_name);
        sb_append_str(&o, " \"");
        sb_append_str(&o, tool_arg);
        sb_append_str(&o, "\":\n");
        sb_append_str(&o, obs);
        free(obs);
        arglist_free(&al);
        msgs_add(&msgs, &nmsg, &capmsg, "user", sb_to_str(&o));
    }

    /* No explicit answer (budget hit or interrupted): synthesize from the full
       chat. Deliver the instruction as/within a trailing user turn. */
    if (!answer) {
        if (!generation_interrupted)
            printf("\033[36m[deepsearch] round budget reached — synthesizing final answer\033[0m\n");
        fflush(stdout);
        const char *synth = "\n\n[Research budget reached. Stop searching. Based ONLY on everything "
                            "above, give your final, thorough answer now inside <answer>...</answer>, "
                            "citing the source URLs your findings recorded.]";
        if (nmsg > 0 && strcmp(msgs[nmsg - 1].role, "user") == 0) {
            size_t ol = strlen(msgs[nmsg - 1].content);
            char *nc = malloc(ol + strlen(synth) + 1);
            if (nc) {   /* on OOM, skip the append and keep the original prompt */
                memcpy(nc, msgs[nmsg - 1].content, ol);
                strcpy(nc + ol, synth);
                free((void *)msgs[nmsg - 1].content);
                msgs[nmsg - 1].content = nc;
            }
        } else {
            msgs_add(&msgs, &nmsg, &capmsg, "user",
                     strdup("[Research budget reached. Give your final, thorough, cited <answer> now.]"));
        }
        generation_interrupted = 0;   /* let the synthesis run even after a Ctrl+C */
        char *out = ds_generate(dctx, vocab, dsmpl, model, msgs, nmsg, fmt_buf);
        if (out) {
            /* Prefer <answer>; the 4B sometimes emits <report> instead — use that
               body then, never raw tagged text. */
            const char *fact = after_think(out);
            char *ans = extract_tagged(fact, "answer");
            if (!ans) ans = extract_tagged(fact, "report");
            answer = ans ? (trim(ans), ans) : strdup(fact);  /* never surface <think> in the result */
            free(out);
        }
    }

    if (!answer || !answer[0]) {
        free(answer);
        answer = strdup("deep_search produced no final answer.");
    }

    /* Teardown — never free the model (owned by main). */
    sigaction(SIGINT, &old_sa, NULL);
    generation_interrupted = 0;
    generate_quiet = prev_quiet;
    generate_keep_think = prev_keep_think;
    basi_srv_suppress_grammar = prev_suppress_grammar;
    for (size_t i = 0; i < nmsg; i++) free((void *)msgs[i].content);
    free(msgs);
    free(fmt_buf);
    llama_sampler_free(xsmpl);
    llama_sampler_free(dsmpl);
    llama_free(dctx);

    return answer;
}
