#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <termios.h>
#include <poll.h>
#include <signal.h>
#include <time.h>
#include <errno.h>

#include "basi_types.h"

#include "util.h"
#include "globals.h"
#include "model.h"
#include "hwinfo.h"
#include "chat_tmpl.h"
#include "md.h"
#include "srvgen.h"
#include "srvchat.h"
#include "bestof.h"
#include "backend.h"   /* the BACKEND row's candidates */
#include "vramobs.h"   /* measured VRAM, when this model has launched before */


/* ── Spinner frames ────────────────────────────────────────────────── */

static const char *spinner_frames[] = {
    "\xe2\xa0\x8b", "\xe2\xa0\x99", "\xe2\xa0\xb9", "\xe2\xa0\xb8",
    "\xe2\xa0\xbc", "\xe2\xa0\xb4", "\xe2\xa0\xa6", "\xe2\xa0\xa7",
    "\xe2\xa0\x87", "\xe2\xa0\x8f"
};
#define SPINNER_COUNT 10
/* ── Extract <tool>...</tool> from text ────────────────────────────── */

const char *extract_tool_call(const char *text, size_t *out_len) {
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
    if (generate_quiet) return;
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
    if (generate_quiet) return;
    printf("\r\033[K\n\033[K\n\033[K\033[2A\r");
    fflush(stdout);
}


/* ── Generate response ─────────────────────────────────────────────── */


/* ── Server-backed generation (Pi-style) ─────────────────────────────────────
 * When basi_srv_port>0, generate() delegates token generation to a spawned
 * llama-server over its /completion SSE stream instead of decoding in-process.
 * Set by main.c after srvgen_spawn(). basi_srv_model is a vocab_only handle used
 * to derive the tool grammar. */
int basi_srv_port = 0;
/* Defaults mirror BASI's native sampler chain (main.c overwrites in server mode):
   temp 0.4, repeat_penalty 1.1 over 256 tokens, min_p 0.05, top_k/top_p/seed off. */
SrvSampling basi_srv_sampling = { .temperature = 0.4, .repeat_penalty = 1.1, .repeat_last_n = 256,
                                  .min_p = 0.05, .top_k = 0, .top_p = 1.0, .seed = -1 };
/* When set, generate_server() omits the tool-call grammar. deepsearch drives its
   own ReAct loop with a private sampler chain that (in native mode) carries no
   grammar; this flag reproduces that on the server path so the main tool grammar
   can't leak into deepsearch's rounds or its final synthesis. */
int basi_srv_suppress_grammar = 0;
/* When set, build_request() adds chat_template_kwargs.enable_thinking=false to
   every chat-completions request — the per-request equivalent of BASI_NO_THINK,
   scoped by a caller that sets it around a sub-loop. study_ground sets it around
   its ReAct grounding turns: a Gemma-4 / Qwen3.x model then investigates the
   codebase WITHOUT a multi-thousand-token <think> per tool call, while the
   hypothesis step (which does NOT set it) keeps full reasoning. Not process-wide:
   it is a scoped toggle, restored by the caller — never a global setenv. */
int basi_srv_no_think = 0;


/* ── Chat-completions path (item 6b): server templates from messages, owns the
 * tool grammar, and returns STRUCTURED tool_calls + separated reasoning. The
 * display is simpler than the /completion state machine — reasoning arrives on a
 * distinct stream, so no <think> tag parsing: reasoning → thinking box (or dim
 * text under Ctrl+T), content → the markdown answer. ─────────────────────────── */
typedef struct {
    int    md;
    bool   thinking_box_shown;
    bool   md_started;
    size_t spinner_frame;
    double last_spinner;
} ChatDisplay;

static void chat_on_reasoning(const char *chunk, void *ud) {
    ChatDisplay *d = (ChatDisplay *) ud;
    if (generate_quiet) return;
    if (show_thinking) {
        printf("\033[90m%s\033[0m", chunk); fflush(stdout);
        d->thinking_box_shown = true;   /* so a newline is emitted when it closes */
    } else {
        double now = time_now();
        if (!d->thinking_box_shown || now - d->last_spinner > 0.08) {
            d->spinner_frame++; draw_thinking_box(d->spinner_frame); d->last_spinner = now;
            d->thinking_box_shown = true;
        }
    }
}

static void chat_on_content(const char *chunk, void *ud) {
    ChatDisplay *d = (ChatDisplay *) ud;
    if (generate_quiet) return;
    if (d->thinking_box_shown) {   /* reasoning finished — tear the box down first */
        if (show_thinking) printf("\033[0m\n"); else clear_thinking_box();
        d->thinking_box_shown = false;
    }
    if (d->md) {
        if (!d->md_started) { md_begin(); d->md_started = true; }   /* open the answer stream once */
        md_feed(chunk, strlen(chunk));
    } else { printf("\033[0m"); fputs(chunk, stdout); fflush(stdout); }
}

/* Server-chat generation. Serializes `messages` (+ registered tools) to OpenAI
 * JSON, streams /v1/chat/completions, and returns the answer text plus STRUCTURED
 * tool calls in tc_out/n_tc_out (caller frees via basi_free_tool_calls).
 * res.prompt_tokens carries the server's exact prompt count for ctx accounting. */
GenerateResult generate_chat(const BasiMsg *messages, size_t msg_count,
                             BasiToolCall **tc_out, int *n_tc_out) {
    GenerateResult res = { NULL, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0 };
    if (tc_out) *tc_out = NULL;
    if (n_tc_out) *n_tc_out = 0;

    char *mj = basi_messages_to_json(messages, (int) msg_count);
    if (!mj) { res.text = strdup("[chat serialization failed]"); return res; }
    /* deepsearch clears g_tools for its own ReAct loop → basi_tools_to_json()
       returns NULL there, which is exactly what we want (no tools advertised). */
    char *tj = basi_tools_to_json();

    long cap = 0; { const char *e = getenv("BASI_MAX_TOKENS"); if (e && *e) cap = atol(e); }
    int n_predict = cap > 0 ? (int) cap : -1;

    ChatDisplay disp = { (generate_markdown && !generate_quiet) ? 1 : 0, false, false, 0, 0.0 };
    double t0 = time_now();

    /* BASI_BEST_OF_N>1: sample N turns from ONE prefill and keep the consensus
       pick. Costs ~1.95x a single turn for N=4 (not Nx) because the server
       prefills once — but it REQUIRES the launch script to carry `-np N
       --kv-unified`, or the server caps n at the slot count and 400s the request.
       Streaming is suppressed here: N interleaved token streams are unreadable,
       so the turn renders after the winner is chosen. */
    int best_of = 0;
    { const char *e = getenv("BASI_BEST_OF_N"); if (e && *e) best_of = atoi(e); }

    /* Whether the answer was already shown live by the streaming callbacks. The
       best-of-N path streams nothing (N interleaved streams are unreadable), so
       its winner has to be rendered explicitly once chosen. */
    bool streamed = true;

    SrvChatResult *r = NULL;
    if (best_of > 1) {
        SrvChatResult **cands = calloc((size_t) best_of, sizeof *cands);
        int got = cands ? srvchat_complete_n(basi_srv_port, mj, tj, &basi_srv_sampling,
                                             n_predict, best_of, NULL, NULL, NULL,
                                             cands, best_of)
                        : -1;
        if (got > 0) {
            BestOfPick pick = bestof_select(cands, got);
            /* winner<0 means NOTHING was usable — every candidate empty, which is
               exactly what a server-rejected request looks like (an error response
               carries no deltas, so each result parses to empty rather than NULL).
               Taking cands[0] here would hand back a blank turn and suppress the
               fallback; bestof.h's contract is to fall through to a single sample. */
            if (pick.winner < 0) {
                if (!generate_quiet)
                    fprintf(stderr, "\033[33m[best-of-%d] no usable candidate "
                            "(server rejected the request?) — retrying single-sample\033[0m\n", best_of);
                for (int i = 0; i < got; i++) srvchat_free(cands[i]);
                bestof_free(&pick);
                free(cands);
                cands = NULL;
                goto single_sample;
            }
            int w = pick.winner;
            if (!generate_quiet)
                fprintf(stderr, "\033[90m[best-of-%d] %s\033[0m\n", got,
                        pick.reason ? pick.reason : "no consensus, kept first");
            /* BASI_BEST_OF_DEBUG=1 shows the REJECTED candidates too. A ranker you
               can't see the losers of is a ranker you can't evaluate — especially
               in medoid mode, where "most central" is only a proxy for "best". */
            if (getenv("BASI_BEST_OF_DEBUG")) {
                for (int i = 0; i < got; i++) {
                    if (!cands[i]) continue;
                    const char *c = cands[i]->content ? cands[i]->content : "";
                    if (cands[i]->n_tool_calls > 0)
                        fprintf(stderr, "\033[90m  %s [%d] %s(%.90s)\033[0m\n",
                                i == w ? "->" : "  ", i,
                                cands[i]->tool_calls[0].name ? cands[i]->tool_calls[0].name : "?",
                                cands[i]->tool_calls[0].arguments ? cands[i]->tool_calls[0].arguments : "");
                    else
                        fprintf(stderr, "\033[90m  %s [%d] %zub: %.140s…\033[0m\n",
                                i == w ? "->" : "  ", i, strlen(c), c);
                }
            }
            r = cands[w];
            streamed = false;
            for (int i = 0; i < got; i++) if (i != w) srvchat_free(cands[i]);
            bestof_free(&pick);
        }
        free(cands);
        /* Fall through to the single-sample path if the N-way request failed. */
    }
single_sample:
    if (!r)
        r = srvchat_complete(basi_srv_port, mj, tj, &basi_srv_sampling, n_predict,
                             chat_on_content, chat_on_reasoning, &disp);
    free(mj); free(tj);

    if (!r) {
        if (disp.thinking_box_shown && !generate_quiet) { if (show_thinking) printf("\033[0m\n"); else clear_thinking_box(); }
        if (disp.md && disp.md_started) md_end();
        res.text = strdup("[server chat failed]");
        return res;
    }

    if (getenv("BASI_DEBUG_CHAT"))
        fprintf(stderr, "\n[chat] finish=%s content=%zub reasoning=%zub tool_calls=%d\n  reasoning=[%.400s]\n  content=[%.200s]\n",
                r->finish_reason ? r->finish_reason : "?",
                r->content ? strlen(r->content) : 0,
                r->reasoning ? strlen(r->reasoning) : 0, r->n_tool_calls,
                r->reasoning ? r->reasoning : "", r->content ? r->content : "");

    /* Qwen3.x sometimes keeps a brief final answer INSIDE the reasoning stream
       (thinks the answer, closes, stops) → content comes back empty. When there's
       no answer content and no tool call, promote the reasoning to the answer so
       the turn isn't blank; it was shown live as a hidden box, so reveal it now. */
    bool answer_in_reasoning = (!r->content || !r->content[0]) && r->n_tool_calls == 0
                               && r->reasoning && r->reasoning[0];
    const char *answer = answer_in_reasoning ? r->reasoning : (r->content ? r->content : "");

    if (disp.thinking_box_shown && !generate_quiet) {   /* tear the reasoning box down */
        if (show_thinking) printf("\033[0m\n"); else clear_thinking_box();
        disp.thinking_box_shown = false;
    }
    /* Render when the answer was never shown live: either the reasoning box hid
       it, or best-of-N suppressed streaming entirely. */
    if ((answer_in_reasoning || !streamed) && !generate_quiet) {
        if (disp.md) { md_begin(); md_feed(answer, strlen(answer)); md_end(); }
        else { printf("\033[0m"); fputs(answer, stdout); }
    } else if (disp.md && disp.md_started) {
        md_end();                                        /* close the answer stream */
    }
    if (!generate_quiet) { printf("\033[0m\n"); fflush(stdout); }

    res.text          = strdup(answer);
    res.prompt_tokens    = (size_t) r->prompt_tokens;
    res.gen_tokens       = (size_t) r->completion_tokens;
    res.cached_tokens    = (size_t) r->cached_tokens;
    res.reasoning_tokens = (size_t) r->reasoning_tokens;
    res.gen_time_s    = (r->tps > 0) ? r->completion_tokens / r->tps : (time_now() - t0);
    /* Prefill time comes straight from the server's own clock. The old form,
       prompt_tokens/prompt_tps, divided the whole prompt by a rate measured over
       only the evaluated tokens; on a cache-hit turn (prompt_tokens=15442,
       prompt_n=19) that reported ~400s of prefill for 0.49s of work. */
    res.prompt_n      = (size_t) r->prompt_n;
    res.prompt_time_s = r->prompt_ms / 1000.0;
    res.prompt_tps    = r->prompt_tps;

    /* BASI_TIMING_TRACE=1: one TSV row per model round, so the prefill/decode split
       of a real agentic run can be summed from the server's clock instead of
       estimated. Columns: prompt_tokens (occupancy) prompt_n (work) prefill_s
       gen_tokens gen_s. */
    if (getenv("BASI_TIMING_TRACE")) {
        fprintf(stderr, "[timing]\t%zu\t%zu\t%.3f\t%zu\t%.3f\n",
                res.prompt_tokens, res.prompt_n, res.prompt_time_s,
                res.gen_tokens, res.gen_time_s);
        fflush(stderr);
    }

    if (r->n_tool_calls > 0 && tc_out) {
        BasiToolCall *arr = calloc((size_t) r->n_tool_calls, sizeof(BasiToolCall));
        if (arr) {
            for (int i = 0; i < r->n_tool_calls; i++) {
                arr[i].name      = strdup(r->tool_calls[i].name      ? r->tool_calls[i].name      : "");
                arr[i].arguments = strdup(r->tool_calls[i].arguments ? r->tool_calls[i].arguments : "{}");
            }
            *tc_out = arr;
            if (n_tc_out) *n_tc_out = r->n_tool_calls;
        }
    }
    srvchat_free(r);
    return res;
}


/* Chat templating is done server-side now (llama-server templates from the
   messages we POST to /v1/chat/completions), so there is no in-process
   apply_template / basi_render_chat anymore. */

/* ── Model picker ──────────────────────────────────────────────────── */

#define MODEL_DIRS_MAX 4
static const char *model_search_dirs[] = {
    NULL, /* [0] filled at runtime: ~/.cache/huggingface/hub */
    NULL, /* [1] filled at runtime: $HOME/models (where /cookbook downloads) */
    ".",
    NULL
};

/* Resolve the two $HOME-relative search dirs at runtime: [0] the HF hub cache,
 * [1] the ~/models dir /cookbook downloads into. No-op if $HOME is unset. */
static void init_model_search_dirs(void) {
    static char cache_dir[512];
    static char models_dir[512];
    const char *home = getenv("HOME");
    if (!home) return;
    snprintf(cache_dir,  sizeof(cache_dir),  "%s/.cache/huggingface/hub", home);
    snprintf(models_dir, sizeof(models_dir), "%s/models", home);
    model_search_dirs[0] = cache_dir;
    model_search_dirs[1] = models_dir;
}

/* GGUF arch metadata used to size offload + KV cache. */
typedef struct {
    int n_ctx_train;   /* <arch>.context_length         */
    int n_layers;      /* <arch>.block_count            */
    int n_embd;        /* <arch>.embedding_length       */
    int n_head;        /* <arch>.attention.head_count   */
    int n_head_kv;     /* <arch>.attention.head_count_kv scalar (0 if array/absent) */

    /* Explicit KV head dimensions. When present (e.g. Gemma, DeepSeek) these
     * override the head_dim = n_embd/n_head assumption, which is otherwise
     * wrong. 0 means "fall back to n_embd/n_head". */
    int key_length;          /* <arch>.attention.key_length        */
    int value_length;        /* <arch>.attention.value_length      */
    int key_length_swa;      /* …key_length_swa   (sliding-window layers) */
    int value_length_swa;    /* …value_length_swa */
    int sliding_window;      /* …sliding_window — 0 means no SWA   */

    /* Per-layer arrays (size n_layers) when the GGUF stores them as arrays —
     * Gemma varies KV heads and local/global attention per layer. NULL when
     * the model uses uniform scalars. */
    int  *head_kv_per_layer; /* …head_count_kv as an array */
    bool *is_swa_per_layer;  /* …sliding_window_pattern (true = windowed) */

    /* Exact per-layer weight bytes from walking the tensor table. NULL if
     * tensor walk failed; estimator falls back to file_size/n_layers. */
    double *layer_weight_mb;     /* size n_layers */
    double  fixed_weight_mb;     /* token_embd, output, output_norm, etc. */

    /* Mixture-of-experts. The experts are the bulk of an MoE's weights but only
     * a few are active per token, so pinning them to system RAM (--cpu-moe) and
     * putting every ATTENTION layer on the GPU beats offloading whole layers.
     * Measured on Qwen3.6-35B-A3B: --cpu-moe -ngl 99 gives 33.97 tok/s in
     * ~4.0 GB VRAM, versus 24.60 tok/s in ~5.5 GB for the -ngl 7 that
     * whole-layer fitting picks (p=0.0079). Faster AND smaller. */
    int     n_experts;           /* <arch>.expert_count; 0 = dense */
    double *layer_expert_mb;     /* size n_layers; expert bytes only, subset of layer_weight_mb */
} GGUFArch;

/* True when `key` ends with `suffix` — avoids the substring trap where
 * ".embedding_length" also matches ".embedding_length_per_layer_input". */
static bool key_suffix_is(const char *key, const char *suffix) {
    size_t kl = strlen(key), sl = strlen(suffix);
    return kl >= sl && strcmp(key + kl - sl, suffix) == 0;
}

/* Bytes per element for a given GGML tensor dtype. Returns 4.0 for unknown
 * types so we don't grossly underestimate. Block sizes match ggml-quants. */
static double ggml_dtype_bytes_per_elem(uint32_t dtype) {
    switch (dtype) {
    case 0:  return 4.0;             /* F32   */
    case 1:  return 2.0;             /* F16   */
    case 2:  return 18.0 / 32.0;     /* Q4_0  */
    case 3:  return 20.0 / 32.0;     /* Q4_1  */
    case 6:  return 22.0 / 32.0;     /* Q5_0  */
    case 7:  return 24.0 / 32.0;     /* Q5_1  */
    case 8:  return 34.0 / 32.0;     /* Q8_0  */
    case 9:  return 40.0 / 32.0;     /* Q8_1  */
    case 10: return 84.0 / 256.0;    /* Q2_K  */
    case 11: return 110.0 / 256.0;   /* Q3_K  */
    case 12: return 144.0 / 256.0;   /* Q4_K  */
    case 13: return 176.0 / 256.0;   /* Q5_K  */
    case 14: return 210.0 / 256.0;   /* Q6_K  */
    case 15: return 292.0 / 256.0;   /* Q8_K  */
    case 16: return 66.0  / 256.0;   /* IQ2_XXS */
    case 17: return 74.0  / 256.0;   /* IQ2_XS  */
    case 18: return 110.0 / 256.0;   /* IQ3_XXS */
    case 19: return 50.0  / 256.0;   /* IQ1_S   */
    case 20: return 18.0  / 32.0;    /* IQ4_NL  */
    case 21: return 110.0 / 256.0;   /* IQ3_S   */
    case 22: return 82.0  / 256.0;   /* IQ2_S   */
    case 23: return 136.0 / 256.0;   /* IQ4_XS  */
    case 24: return 1.0;             /* I8      */
    case 25: return 2.0;             /* I16     */
    case 26: return 4.0;             /* I32     */
    case 27: return 8.0;             /* I64     */
    case 28: return 8.0;             /* F64     */
    case 29: return 56.0  / 256.0;   /* IQ1_M   */
    case 30: return 2.0;             /* BF16    */
    /* Q4_0_4_4 / Q4_0_4_8 / Q4_0_8_8 share Q4_0 layout. */
    case 31: case 32: case 33: return 18.0 / 32.0;
    case 34: return 26.0 / 256.0;    /* TQ1_0 */
    case 35: return 64.0 / 256.0;    /* TQ2_0 */
    default: return 4.0;             /* unknown — overestimate slightly */
    }
}

/* Parse a tensor name like "blk.13.ffn_down_exps.weight" → 13.
 * Returns -1 for non-block tensors (token_embd, output, output_norm, …). */
static int tensor_layer_index(const char *name) {
    if (strncmp(name, "blk.", 4) != 0) return -1;
    int idx = 0;
    const char *p = name + 4;
    if (*p < '0' || *p > '9') return -1;
    while (*p >= '0' && *p <= '9') { idx = idx * 10 + (*p - '0'); p++; }
    return idx;
}

/* Read GGUF metadata fields needed for VRAM estimation. Missing fields stay 0. */
static GGUFArch read_gguf_arch(const char *path) {
    GGUFArch r = {0};
    FILE *f = fopen(path, "rb");
    if (!f) return r;

    /* GGUF header: magic(4) + version(4) + tensor_count(8) + metadata_count(8) */
    uint32_t magic, version;
    uint64_t tensor_count, metadata_count;
    if (fread(&magic, 4, 1, f) != 1) { fclose(f); return r; }
    if (fread(&version, 4, 1, f) != 1) { fclose(f); return r; }
    if (fread(&tensor_count, 8, 1, f) != 1) { fclose(f); return r; }
    if (fread(&metadata_count, 8, 1, f) != 1) { fclose(f); return r; }

    /* Per-layer arrays captured during the metadata walk; mapped to n_layers
     * after the loop (metadata order isn't guaranteed). */
    int     *tmp_kv     = NULL; uint64_t tmp_kv_len  = 0;
    bool    *tmp_swa    = NULL; uint64_t tmp_swa_len = 0;
    const uint64_t ARR_CAP = 4096;  /* sane upper bound on layer count */

    bool metadata_clean = true;
    for (uint64_t i = 0; i < metadata_count; i++) {
        /* key: len(8) + data */
        uint64_t key_len;
        if (fread(&key_len, 8, 1, f) != 1) { metadata_clean = false; break; }
        char key[257] = {0};
        if (key_len > 256) {
            /* Skip the entire over-sized key; we don't care about its name. */
            if (fseek(f, (long)key_len, SEEK_CUR) != 0) { metadata_clean = false; break; }
        } else {
            if (fread(key, 1, key_len, f) != key_len) { metadata_clean = false; break; }
            key[key_len] = '\0';
        }

        /* value type */
        uint32_t vtype;
        if (fread(&vtype, 4, 1, f) != 1) { metadata_clean = false; break; }

        if (vtype == 4 || vtype == 5) { /* uint32 / int32 */
            uint32_t val;
            if (fread(&val, 4, 1, f) != 1) { metadata_clean = false; break; }
            /* Exact-suffix matching: ".embedding_length" must NOT match
             * ".embedding_length_per_layer_input" (=0), which would zero
             * n_embd and silently drop the entire KV-cache estimate. */
            if      (key_suffix_is(key, ".context_length"))             r.n_ctx_train = (int)val;
            else if (key_suffix_is(key, ".block_count"))                r.n_layers    = (int)val;
            else if (key_suffix_is(key, ".embedding_length"))           r.n_embd      = (int)val;
            else if (key_suffix_is(key, ".attention.head_count_kv"))    r.n_head_kv   = (int)val;
            else if (key_suffix_is(key, ".attention.head_count"))       r.n_head      = (int)val;
            else if (key_suffix_is(key, ".attention.key_length_swa"))   r.key_length_swa   = (int)val;
            else if (key_suffix_is(key, ".attention.value_length_swa")) r.value_length_swa = (int)val;
            else if (key_suffix_is(key, ".attention.key_length"))       r.key_length   = (int)val;
            else if (key_suffix_is(key, ".attention.value_length"))     r.value_length = (int)val;
            else if (key_suffix_is(key, ".attention.sliding_window"))   r.sliding_window = (int)val;
            else if (key_suffix_is(key, ".expert_count"))               r.n_experts   = (int)val;
        } else if (vtype == 6) { fseek(f, 4, SEEK_CUR); }
        else if (vtype == 0 || vtype == 1 || vtype == 7) { fseek(f, 1, SEEK_CUR); }
        else if (vtype == 2 || vtype == 3) { fseek(f, 2, SEEK_CUR); }
        else if (vtype == 10 || vtype == 12) { fseek(f, 8, SEEK_CUR); }
        else if (vtype == 8) { /* string */
            uint64_t slen;
            if (fread(&slen, 8, 1, f) != 1) { metadata_clean = false; break; }
            fseek(f, (long)slen, SEEK_CUR);
        } else if (vtype == 9) { /* array */
            uint32_t atype;
            uint64_t alen;
            if (fread(&atype, 4, 1, f) != 1) { metadata_clean = false; break; }
            if (fread(&alen, 8, 1, f) != 1)  { metadata_clean = false; break; }

            /* Capture the two per-layer arrays Gemma stores as arrays: KV head
             * counts (int) and the local/global attention pattern (bool). */
            int  *cap_kv  = NULL;
            bool *cap_swa = NULL;
            if (alen > 0 && alen <= ARR_CAP) {
                if ((atype == 4 || atype == 5) &&
                    key_suffix_is(key, ".attention.head_count_kv")) {
                    cap_kv = calloc((size_t)alen, sizeof(int));
                } else if (atype == 7 &&
                           key_suffix_is(key, ".attention.sliding_window_pattern")) {
                    cap_swa = calloc((size_t)alen, sizeof(bool));
                }
            }

            bool array_clean = true;
            for (uint64_t a = 0; a < alen; a++) {
                if (atype == 8) {
                    uint64_t slen;
                    if (fread(&slen, 8, 1, f) != 1) { array_clean = false; break; }
                    fseek(f, (long)slen, SEEK_CUR);
                } else if (atype == 4 || atype == 5 || atype == 6) {
                    uint32_t v;
                    if (fread(&v, 4, 1, f) != 1) { array_clean = false; break; }
                    if (cap_kv) cap_kv[a] = (int)v;
                } else if (atype == 0 || atype == 1 || atype == 7) {
                    uint8_t v;
                    if (fread(&v, 1, 1, f) != 1) { array_clean = false; break; }
                    if (cap_swa) cap_swa[a] = (v != 0);
                } else if (atype == 2 || atype == 3) { fseek(f, 2, SEEK_CUR); }
                else if (atype == 10 || atype == 12) { fseek(f, 8, SEEK_CUR); }
                else { array_clean = false; break; }
            }
            if (!array_clean) {
                free(cap_kv); free(cap_swa);
                metadata_clean = false; break;
            }
            if (cap_kv)  { free(tmp_kv);  tmp_kv  = cap_kv;  tmp_kv_len  = alen; }
            if (cap_swa) { free(tmp_swa); tmp_swa = cap_swa; tmp_swa_len = alen; }
        } else {
            metadata_clean = false; /* unknown type — file position is now lost */
            break;
        }
    }

    /* Map captured per-layer arrays onto n_layers (clamp if lengths differ). */
    if (r.n_layers > 0) {
        if (tmp_kv && tmp_kv_len > 0) {
            r.head_kv_per_layer = calloc((size_t)r.n_layers, sizeof(int));
            if (r.head_kv_per_layer)
                for (uint64_t i = 0; i < (uint64_t)r.n_layers; i++)
                    r.head_kv_per_layer[i] = tmp_kv[i < tmp_kv_len ? i : tmp_kv_len - 1];
        }
        if (tmp_swa && tmp_swa_len > 0) {
            r.is_swa_per_layer = calloc((size_t)r.n_layers, sizeof(bool));
            if (r.is_swa_per_layer)
                for (uint64_t i = 0; i < (uint64_t)r.n_layers; i++)
                    r.is_swa_per_layer[i] = tmp_swa[i < tmp_swa_len ? i : tmp_swa_len - 1];
        }
    }
    free(tmp_kv);
    free(tmp_swa);

    /* Walk tensor info section: each entry is name(str) + n_dims(u32)
     * + dims[n_dims](u64) + dtype(u32) + offset(u64). Exact per-layer
     * weight bytes matter for MoE models where layers are wildly uneven. */
    if (metadata_clean && r.n_layers > 0 && tensor_count > 0 && tensor_count < 1000000) {
        r.layer_weight_mb = calloc((size_t)r.n_layers, sizeof(double));
        r.layer_expert_mb = calloc((size_t)r.n_layers, sizeof(double));
        if (r.layer_weight_mb && r.layer_expert_mb) {
            bool tensor_clean = true;
            for (uint64_t t = 0; t < tensor_count; t++) {
                uint64_t name_len;
                if (fread(&name_len, 8, 1, f) != 1) { tensor_clean = false; break; }
                if (name_len > 1024) { tensor_clean = false; break; }
                char name[1025] = {0};
                if (fread(name, 1, name_len, f) != name_len) { tensor_clean = false; break; }
                name[name_len] = '\0';

                uint32_t n_dims;
                if (fread(&n_dims, 4, 1, f) != 1) { tensor_clean = false; break; }
                if (n_dims == 0 || n_dims > 8) { tensor_clean = false; break; }

                uint64_t elements = 1;
                for (uint32_t d = 0; d < n_dims; d++) {
                    uint64_t dim;
                    if (fread(&dim, 8, 1, f) != 1) { tensor_clean = false; break; }
                    elements *= dim;
                }
                if (!tensor_clean) break;

                uint32_t dtype;
                if (fread(&dtype, 4, 1, f) != 1) { tensor_clean = false; break; }
                uint64_t off;
                if (fread(&off, 8, 1, f) != 1)   { tensor_clean = false; break; }

                double mb = (double)elements * ggml_dtype_bytes_per_elem(dtype)
                            / (1024.0 * 1024.0);
                int layer = tensor_layer_index(name);
                if (layer >= 0 && layer < r.n_layers) {
                    r.layer_weight_mb[layer] += mb;
                    /* ffn_{gate,down,up}_exps are the expert stacks — the part
                     * --cpu-moe keeps in system RAM. Tracked separately so the
                     * VRAM estimate can exclude them. */
                    if (strstr(name, "_exps")) r.layer_expert_mb[layer] += mb;
                } else {
                    r.fixed_weight_mb += mb;
                }
            }
            if (!tensor_clean) {
                free(r.layer_weight_mb);
                r.layer_weight_mb = NULL;
                free(r.layer_expert_mb);
                r.layer_expert_mb = NULL;
                r.fixed_weight_mb = 0.0;
            }
        }
    }

    fclose(f);
    return r;
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

/* Public: scan the model search dirs and return every .gguf path found
 * (malloc'd array of malloc'd strings). Caller frees each entry then the array.
 * Returns the count (0 with *out=NULL if none). Used by /model to resolve a
 * substring like "qwen3.6" to a concrete path without the full picker TUI. */
int basi_list_models(char ***out) {
    init_model_search_dirs();
    char **models = NULL;
    int count = 0, cap = 0;
    for (int d = 0; d < MODEL_DIRS_MAX && model_search_dirs[d]; d++)
        scan_gguf_recursive(model_search_dirs[d], &models, &count, &cap);
    *out = models;
    return count;
}

/* Settings values for ←/→ adjustment.
 * GPU_LAYER_AUTO (-1) means: auto-fit to available VRAM each render.
 * Manual values step by 1 in [0, model's n_layers]. */
#define GPU_LAYER_AUTO (-1)
static const float temp_opts[]     = { 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f };
#define N_TEMP_OPTS (int)(sizeof(temp_opts)/sizeof(temp_opts[0]))
#define CTX_STEP    1024  /* slider step size */
#define CTX_MIN     1024
#define CTX_DEFAULT 32768

/* Base Vulkan allocator + command pool + small fixed buffers.
 * Compute scratch on top of this is computed dynamically from ctx × n_embd. */
#define VRAM_OVERHEAD_BASE_MB 600.0
/* Fallback only: when the driver can't report live VRAM budget
 * (VK_EXT_memory_budget absent), assume this much is held by the
 * display/compositor and subtract it from the total heap size. */
#define VRAM_RESERVE_MB 2048.0
/* Safety margin held back from the driver's live free-VRAM figure, to absorb
 * allocator fragmentation and growth during a long generation. */
#define VRAM_SAFETY_MB 512.0

/* Memory split between VRAM and system RAM for a given offload setting. */
typedef struct { double vram_mb; double ram_mb; } MemorySplit;

/* Estimate weights + KV cache split given the model arch and offload count.
 * Uses exact per-layer tensor bytes when the GGUF tensor walk succeeded;
 * otherwise falls back to file_size / n_layers. */
static MemorySplit estimate_memory(double file_size_mb, GGUFArch arch,
                                   int gpu_layers, int ctx, bool cpu_moe) {
    MemorySplit s = {0, 0};
    int total_layers = arch.n_layers > 0 ? arch.n_layers : 32;
    if (gpu_layers > total_layers) gpu_layers = total_layers;
    if (gpu_layers < 0) gpu_layers = 0;

    /* Weights — exact per-layer if the tensor walker succeeded, else even split. */
    if (arch.layer_weight_mb) {
        for (int i = 0; i < gpu_layers; i++) {
            double w = arch.layer_weight_mb[i];
            /* --cpu-moe pins the expert stacks to system RAM even for layers that
             * are otherwise offloaded, so only the attention/norm remainder of an
             * offloaded layer lands in VRAM. */
            if (cpu_moe && arch.layer_expert_mb) {
                s.ram_mb  += arch.layer_expert_mb[i];
                w         -= arch.layer_expert_mb[i];
                if (w < 0) w = 0;
            }
            s.vram_mb += w;
        }
        for (int i = gpu_layers; i < total_layers; i++) s.ram_mb += arch.layer_weight_mb[i];
        /* Fixed tensors (token_embd, output, output_norm) live on CPU when not
         * fully offloaded. llama.cpp puts them on GPU only when n_gpu_layers
         * exceeds n_blocks, which is out of our slider's range. */
        s.ram_mb += arch.fixed_weight_mb;
    } else {
        double weights_per_layer = file_size_mb / (double)total_layers;
        s.vram_mb += gpu_layers * weights_per_layer;
        s.ram_mb  += (total_layers - gpu_layers) * weights_per_layer;
    }

    /* KV cache — fp16, summed exactly per layer. KV head count and the cached
     * token window can vary across layers: Gemma alternates local sliding-
     * window layers (which cache only `sliding_window` tokens, with smaller
     * head dims) against full-context global layers, and uses explicit
     * key/value head dimensions that differ from n_embd/n_head. */
    {
        int n_head    = arch.n_head    > 0 ? arch.n_head    : 1;
        int kdim_full = arch.key_length   > 0 ? arch.key_length
                        : (arch.n_embd > 0 ? arch.n_embd / n_head : 0);
        int vdim_full = arch.value_length > 0 ? arch.value_length
                        : (arch.n_embd > 0 ? arch.n_embd / n_head : 0);
        int kdim_swa  = arch.key_length_swa   > 0 ? arch.key_length_swa   : kdim_full;
        int vdim_swa  = arch.value_length_swa > 0 ? arch.value_length_swa : vdim_full;
        int scalar_kv = arch.n_head_kv > 0 ? arch.n_head_kv : n_head;

        if (kdim_full > 0) {
            for (int i = 0; i < total_layers; i++) {
                int kvh = arch.head_kv_per_layer ? arch.head_kv_per_layer[i] : scalar_kv;
                if (kvh <= 0) kvh = scalar_kv;
                bool swa = arch.is_swa_per_layer ? arch.is_swa_per_layer[i] : false;
                int tokens = ctx, kdim = kdim_full, vdim = vdim_full;
                if (swa && arch.sliding_window > 0) {
                    if (arch.sliding_window < tokens) tokens = arch.sliding_window;
                    kdim = kdim_swa;
                    vdim = vdim_swa;
                }
                /* K and V caches: tokens × kv_heads × head_dim × 2 bytes. */
                double layer_kv_mb = (double)tokens * (double)kvh
                                     * ((double)kdim + (double)vdim) * 2.0
                                     / (1024.0 * 1024.0);
                if (i < gpu_layers) s.vram_mb += layer_kv_mb;
                else                s.ram_mb  += layer_kv_mb;
            }
        }
    }

    /* Compute scratch: dominated by ctx × n_embd intermediate tensors during
     * prompt processing. Coefficient of 6 calibrated empirically against
     * Qwen3-Coder-30B-A3B at ctx=262144 on Intel Arc B580 (Vulkan); lower
     * values under-predict at very long contexts and trigger OOM. */
    if (gpu_layers > 0) {
        double n_embd_for_scratch = arch.n_embd > 0 ? (double)arch.n_embd : 2048.0;
        double compute_scratch_mb = (double)ctx * n_embd_for_scratch * 6.0
                                    / (1024.0 * 1024.0);
        s.vram_mb += VRAM_OVERHEAD_BASE_MB + compute_scratch_mb;
    }
    return s;
}

/* Largest gpu_layers (in [0, n_layers]) whose VRAM footprint fits `vram_budget_mb`.
 * Returns 0 if even the smallest non-zero offload spills. */
static int auto_fit_layers(double file_size_mb, GGUFArch arch,
                           int ctx, double vram_budget_mb, bool cpu_moe) {
    int total = arch.n_layers > 0 ? arch.n_layers : 32;
    for (int g = total; g >= 0; g--) {
        MemorySplit s = estimate_memory(file_size_mb, arch, g, ctx, cpu_moe);
        if (s.vram_mb <= vram_budget_mb) return g;
    }
    return 0;
}

/* An MoE is worth running with --cpu-moe whenever the experts actually dominate.
 * Measured on Qwen3.6-35B-A3B: whole-layer fitting picks -ngl 7 (24.60 tok/s,
 * ~5.5GB) while --cpu-moe -ngl 99 gives 33.97 tok/s in ~4.0GB — faster AND
 * smaller, because only a couple of experts are active per token but every
 * offloaded layer must carry ALL of them. */
static bool arch_prefers_cpu_moe(GGUFArch arch) {
    return arch.n_experts > 0 && arch.layer_expert_mb != NULL;
}

double basi_predict_vram_mb(const char *model_path, int ngl, int ctx) {
    if (!model_path || !*model_path) return -1.0;
    GGUFArch a = read_gguf_arch(model_path);
    double result = -1.0;
    if (a.n_layers > 0) {
        /* ngl<0 (or the conventional 99) means every layer; estimate_memory clamps. */
        int eff = ngl < 0 ? a.n_layers : ngl;
        MemorySplit ms = estimate_memory(file_size_mb(model_path), a, eff, ctx,
                                         arch_prefers_cpu_moe(a));
        result = ms.vram_mb;
    }
    free(a.layer_weight_mb);
    free(a.layer_expert_mb);
    free(a.head_kv_per_layer);
    free(a.is_swa_per_layer);
    return result;
}

/* VRAM freed by a just-exited model (e.g. after a /model re-exec) can lag in the
 * driver's live budget for a few hundred ms. Sample the probe until the free
 * figure stops climbing (or a short timeout), so the picker's VRAM math reflects
 * the fully-offloaded GPU rather than a mid-teardown snapshot. Cheap when nothing
 * is changing: the first two reads already agree and it returns after one tick. */
static HwInfo hw_probe_settled(void) {
    HwInfo prev = hw_probe();
    if (!prev.has_gpu || !prev.vram_budget_known) return prev;
    for (int i = 0; i < 16; i++) {                 /* up to ~2.4s */
        usleep(150000);
        HwInfo cur = hw_probe();
        if (!cur.vram_budget_known) return cur;
        long long climbed = (long long)cur.vram_avail_mb - (long long)prev.vram_avail_mb;
        prev = cur;
        if (climbed <= 32) break;                  /* free stopped rising → settled */
    }
    return prev;
}

/*
 * Scan directories for .gguf files, show interactive menu with settings.
 * Returns filled LaunchConfig, or model_path=NULL on cancel.
 */
LaunchConfig pick_model(void) {
    LaunchConfig cfg = { NULL, 99, CONTEXT_SIZE, 0.4f, 0, 0, 0, NULL };

    /* Build search dirs */
    init_model_search_dirs();

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

    /* Read GGUF arch + cache sizes per model */
    GGUFArch *model_arch    = calloc(count, sizeof(GGUFArch));
    double   *model_size_mb = calloc(count, sizeof(double));
    for (int i = 0; i < count; i++) {
        model_arch[i]    = read_gguf_arch(models[i]);
        model_size_mb[i] = file_size_mb(models[i]);
    }

    /* Hardware probe. Wait for VRAM to settle first (a model offloaded by the
       /model re-exec may still be clearing), then let 'r' re-probe on demand. */
    HwInfo hw = hw_probe_settled();

    /* Menu state */
    enum { SECTION_MODEL, SECTION_GPU, SECTION_CTX, SECTION_TEMP,
           SECTION_SPEC, SECTION_FA, SECTION_BACKEND, SECTION_LAUNCH, SECTION_COUNT };
    int section = SECTION_MODEL;

    /* Which llama-server binary to run — the one field of the composed command line
     * this menu never used to offer. Candidates are declared by the user; with fewer
     * than two there is nothing to choose, so the row is hidden entirely. Start on
     * whatever is currently in effect (saved choice, else the Vulkan default). */
    int n_backends = 0;
    const Backend *backends = backend_list(&n_backends);
    int backend_sel = 0;
    for (int i = 0; i < n_backends; i++)
        if (strcmp(backends[i].name, backend_active()->name) == 0) { backend_sel = i; break; }
    const bool show_backend = (n_backends > 1);
    int model_sel = 0;
    int gpu_setting = GPU_LAYER_AUTO;  /* -1 = auto, else absolute layer count */
    int ctx_val = CTX_DEFAULT;     /* free slider value */
    int temp_idx = 4;              /* default: 0.4 */
    /* llama-server launch flags (baked into .basi/run-llama-server.sh). Until the
       user toggles them, they auto-follow the selected model's MTP-ness (an MTP
       head gives lossless spec-decode); spec is forced off for a non-MTP model
       where draft-mtp has nothing to draft. */
    int spec_on = 0, fa_on = 0;
    int spec_touched = 0, fa_touched = 0;

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

        /* Clamp ctx_val to selected model's training context */
        int max_ctx = model_arch[model_sel].n_ctx_train > 0
                      ? model_arch[model_sel].n_ctx_train : 131072;
        if (ctx_val > max_ctx) ctx_val = max_ctx;

        /* Clamp manual setting to current model's layer count */
        int model_max_layers = model_arch[model_sel].n_layers > 0
                               ? model_arch[model_sel].n_layers : 99;
        if (gpu_setting > model_max_layers) gpu_setting = model_max_layers;

        /* Usable VRAM budget for offload. Prefer the driver's live free figure
         * (VK_EXT_memory_budget) — it already excludes the compositor and any
         * other process holding VRAM, so no fixed display reserve is guessed.
         * Fall back to total-minus-reserve only when the driver can't report
         * it. A small safety margin absorbs allocator growth during inference. */
        double vram_usable_mb = 0.0;
        if (hw.has_gpu) {
            if (hw.vram_budget_known)
                vram_usable_mb = (double)hw.vram_avail_mb - VRAM_SAFETY_MB;
            else
                vram_usable_mb = (double)hw.vram_total_mb - VRAM_RESERVE_MB;
            if (vram_usable_mb < 0) vram_usable_mb = 0;
        }

        /* MoE models run faster in less VRAM with --cpu-moe: the experts are the
         * bulk of the weights but only a couple fire per token, so pinning them to
         * RAM and offloading every attention layer beats fitting whole layers. */
        bool cpu_moe = arch_prefers_cpu_moe(model_arch[model_sel]);

        /* Resolve effective GPU layer count (auto-fit when sentinel is selected) */
        int gpu_effective;
        if (gpu_setting == GPU_LAYER_AUTO) {
            gpu_effective = auto_fit_layers(model_size_mb[model_sel],
                                            model_arch[model_sel],
                                            ctx_val, vram_usable_mb, cpu_moe);
        } else {
            gpu_effective = gpu_setting;
        }

        /* GPU layers */
        printf("%s GPU LAYERS    ",
               section == SECTION_GPU ? "\033[1;33m▸" : "  \033[90m");
        if (gpu_setting == GPU_LAYER_AUTO) {
            printf("\033[1mAUTO\033[0m \033[90m(%d / %d)\033[0m",
                   gpu_effective, model_max_layers);
        } else {
            printf("\033[1m%d\033[0m \033[90m/ %d\033[0m",
                   gpu_setting, model_max_layers);
        }
        if (cpu_moe) printf("  \033[36m--cpu-moe\033[0m");
        if (section == SECTION_GPU) printf("  \033[90m← →\033[0m");
        printf("\033[0m\n");

        /* Context size with slider bar */
        {
            printf("%s CONTEXT SIZE  \033[1m%-6d\033[0m",
                   section == SECTION_CTX ? "\033[1;33m▸" : "  \033[90m",
                   ctx_val);

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

        /* MEMORY row: shows VRAM/RAM split for the current settings */
        {
            MemorySplit ms = estimate_memory(model_size_mb[model_sel],
                                             model_arch[model_sel],
                                             gpu_effective, ctx_val, cpu_moe);
            /* Prefer what this model ACTUALLY used last time it launched at this
               config. Display only — auto_fit_layers deliberately keeps budgeting
               against the raw estimate, because the estimate errs high and that
               slack absorbs allocations it doesn't model (a bigger ubatch costs
               hundreds of MiB of compute buffer). Fitting to a measured-tight
               number could start OOMing. */
            int vram_measured = 0;
            ms.vram_mb = vramobs_correct(models[model_sel], gpu_effective, ctx_val,
                                         ms.vram_mb, &vram_measured);
            bool fits_gpu = hw.has_gpu && ms.vram_mb <= vram_usable_mb;
            bool spilling = ms.ram_mb > 0.5;  /* anything not on GPU */

            const char *gpu_color = !hw.has_gpu     ? "\033[90m"
                                  : !fits_gpu       ? "\033[1;31m"   /* red: doesn't fit */
                                  : spilling        ? "\033[1;33m"   /* yellow: partial */
                                                    : "\033[1;32m";  /* green: full GPU */
            const char *ram_color = spilling        ? "\033[1;33m" : "\033[90m";

            printf("    \033[90mMEMORY        \033[0m");
            if (hw.has_gpu) {
                printf("%s[%s %.1f / %.1f GB]\033[0m%s  ",
                       gpu_color, hw_vendor_label(hw.vendor_id),
                       ms.vram_mb / 1024.0, vram_usable_mb / 1024.0,
                       vram_measured ? " \033[32mmeasured\033[0m" : "");
            } else {
                printf("\033[90m[no GPU detected]\033[0m  ");
            }
            printf("%s[RAM %.1f / %.1f GB]\033[0m\n",
                   ram_color, ms.ram_mb / 1024.0,
                   (double)hw.ram_total_mb / 1024.0);
            if (hw.has_gpu && hw.vram_budget_known) {
                printf("    \033[90m%s  %.1f GB free of %.1f GB (live)\033[0m\n",
                       hw.gpu_name, (double)hw.vram_avail_mb / 1024.0,
                       (double)hw.vram_total_mb / 1024.0);
            } else if (hw.has_gpu) {
                printf("    \033[90m%s  (%.1f GB reserved for display, estimated)\033[0m\n",
                       hw.gpu_name, VRAM_RESERVE_MB / 1024.0);
            }
        }

        /* Temperature */
        printf("%s TEMPERATURE   \033[1m%-6.1f\033[0m",
               section == SECTION_TEMP ? "\033[1;33m▸" : "  \033[90m",
               temp_opts[temp_idx]);
        if (section == SECTION_TEMP) printf("  \033[90m← →\033[0m");
        printf("\033[0m\n");

        /* llama-server launch flags — these become the .basi/run-llama-server.sh
           command. Spec-decode needs an MTP head, so it's n/a for non-MTP models. */
        int cur_mtp = (strstr(models[model_sel], "MTP") || strstr(models[model_sel], "mtp")) ? 1 : 0;
        if (!spec_touched) spec_on = cur_mtp;   /* auto-follow model until toggled */
        if (!fa_touched)   fa_on   = cur_mtp;
        if (!cur_mtp) spec_on = 0;              /* draft-mtp needs an MTP head */
        printf("%s SPEC-DECODE   \033[1m%s\033[0m",
               section == SECTION_SPEC ? "\033[1;33m▸" : "  \033[90m",
               !cur_mtp ? "n/a (no MTP head)" : (spec_on ? "draft-mtp (n-max 1)" : "off"));
        if (section == SECTION_SPEC && cur_mtp) printf("  \033[90m← →\033[0m");
        printf("\033[0m\n");

        printf("%s FLASH-ATTN    \033[1m%s\033[0m",
               section == SECTION_FA ? "\033[1;33m▸" : "  \033[90m",
               fa_on ? "on" : "off");
        if (section == SECTION_FA) printf("  \033[90m← →\033[0m");
        printf("\033[0m\n");

        /* BACKEND: which llama-server binary the generated script execs. The extra
           flags come from the backend config, so show them — they are part of the
           command this menu is composing. */
        if (show_backend) {
            printf("%s BACKEND       \033[1m%s\033[0m",
                   section == SECTION_BACKEND ? "\033[1;33m▸" : "  \033[90m",
                   backends[backend_sel].name);
            if (backends[backend_sel].extra_flags[0])
                printf("  \033[36m%s\033[0m", backends[backend_sel].extra_flags);
            if (section == SECTION_BACKEND) printf("  \033[90m← →\033[0m");
            printf("\033[0m\n");
        }

        printf("\n");

        /* Launch button */
        if (section == SECTION_LAUNCH) {
            printf("  \033[1;32m▸ [ LAUNCH ]\033[0m\n");
        } else {
            printf("    \033[90m[ LAUNCH ]\033[0m\n");
        }

        printf("\n\033[90m↑/↓ navigate  ←/→ adjust  r refresh VRAM  Enter select/launch  q quit\033[0m\n");
        fflush(stdout);

        /* Read key */
        unsigned char ch;
        if (read(STDIN_FILENO, &ch, 1) != 1) break;

        if (ch == 'q' || ch == 'Q' || ch == 3) break;

        if (ch == 'r' || ch == 'R') {          /* re-probe live VRAM on demand */
            hw = hw_probe_settled();
            continue;
        }

        if (ch == '\n' || ch == '\r') {
            if (section == SECTION_LAUNCH || section == SECTION_MODEL) {
                if (section == SECTION_MODEL) {
                    /* Enter on model goes to next section */
                    section = SECTION_GPU;
                    continue;
                }
                /* Launch */
                cfg.model_path = strdup(models[model_sel]);
                cfg.gpu_layers = gpu_effective;
                cfg.ctx_size = ctx_val;
                cfg.temperature = temp_opts[temp_idx];
                cfg.spec_draft_mtp = spec_on;
                cfg.flash_attn = fa_on;
                cfg.backend = show_backend ? backends[backend_sel].name : NULL;
                break;
            } else {
                /* Enter on setting goes to next section */
                section++;
                if (section == SECTION_BACKEND && !show_backend) section = SECTION_LAUNCH;
            }
            continue;
        }

        if (ch == 27) {
            unsigned char seq[2];
            if (read(STDIN_FILENO, seq, 2) == 2 && seq[0] == '[') {
                switch (seq[1]) {
                case 'A': /* Up */
                    if (section == SECTION_MODEL && model_sel > 0) model_sel--;
                    else if (section > SECTION_MODEL) {
                        section--;
                        /* Never land on the hidden BACKEND row. */
                        if (section == SECTION_BACKEND && !show_backend) section--;
                    }
                    break;
                case 'B': /* Down */
                    if (section == SECTION_MODEL && model_sel < count - 1) model_sel++;
                    else if (section < SECTION_LAUNCH) {
                        section++;
                        if (section == SECTION_BACKEND && !show_backend) section++;
                    }
                    break;
                case 'C': /* Right */
                    if (section == SECTION_GPU) {
                        int max_l = model_arch[model_sel].n_layers > 0
                                    ? model_arch[model_sel].n_layers : 99;
                        if (gpu_setting < max_l) gpu_setting++;
                    }
                    if (section == SECTION_CTX) {
                        int mc = model_arch[model_sel].n_ctx_train > 0
                                 ? model_arch[model_sel].n_ctx_train : 131072;
                        ctx_val += CTX_STEP;
                        if (ctx_val > mc) ctx_val = mc;
                    }
                    if (section == SECTION_TEMP && temp_idx < N_TEMP_OPTS - 1) temp_idx++;
                    if (section == SECTION_SPEC && cur_mtp) { spec_on = 1; spec_touched = 1; }
                    if (section == SECTION_FA) { fa_on = 1; fa_touched = 1; }
                    if (section == SECTION_BACKEND && backend_sel < n_backends - 1) backend_sel++;
                    break;
                case 'D': /* Left */
                    if (section == SECTION_GPU && gpu_setting > GPU_LAYER_AUTO) gpu_setting--;
                    if (section == SECTION_CTX) {
                        ctx_val -= CTX_STEP;
                        if (ctx_val < CTX_MIN) ctx_val = CTX_MIN;
                    }
                    if (section == SECTION_TEMP && temp_idx > 0) temp_idx--;
                    if (section == SECTION_SPEC) { spec_on = 0; spec_touched = 1; }
                    if (section == SECTION_FA) { fa_on = 0; fa_touched = 1; }
                    if (section == SECTION_BACKEND && backend_sel > 0) backend_sel--;
                    break;
                }
            }
        }
    }

    if (raw) tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig);
    printf("\033[2J\033[H");
    fflush(stdout);

    for (int i = 0; i < count; i++) {
        free(models[i]);
        free(model_arch[i].layer_weight_mb);
        free(model_arch[i].layer_expert_mb);   /* allocated beside layer_weight_mb */
        free(model_arch[i].head_kv_per_layer);
        free(model_arch[i].is_swa_per_layer);
    }
    free(models);
    free(model_arch);
    free(model_size_mb);
    return cfg;
}
