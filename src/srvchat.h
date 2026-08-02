#ifndef BASI_SRVCHAT_H
#define BASI_SRVCHAT_H
/* /v1/chat/completions streaming client — the pure-HTTP path that lets BASI drop
 * the in-process libllama link (item 6). The server templates from `messages`,
 * derives+applies the tool-call grammar from `tools`, separates reasoning from
 * the answer, and returns STRUCTURED tool_calls plus token usage — so BASI no
 * longer needs common_chat templating / grammar / PEG-parsing in-process.
 *
 * Self-contained: only nlohmann/json (header-only) + curl. No llama_/ggml symbols. */
#include "srvgen.h"   /* SrvSampling */
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char *id;          /* server-assigned call id (malloc'd) */
    char *name;        /* function name (malloc'd) */
    char *arguments;   /* function arguments, a JSON object string (malloc'd) */
} SrvToolCall;

typedef struct {
    char        *content;         /* accumulated answer text (malloc'd; may be "") */
    char        *reasoning;       /* accumulated reasoning text (malloc'd; may be NULL) */
    SrvToolCall *tool_calls;      /* array (malloc'd; NULL if none) */
    int          n_tool_calls;
    int          prompt_tokens;   /* from usage: the WHOLE prompt, INCLUDING the KV-cache-hit
                                     prefix. This is context occupancy, NOT work done.
                                     On a hosted API it is also exactly what you are
                                     BILLED for on this turn (the full history is resent). */
    int          completion_tokens;
    /* usage.prompt_tokens_details.cached_tokens — the part of prompt_tokens that hit
       the provider's prompt cache, normally billed at a discount. 0 when absent.
       A SUBSET of prompt_tokens, never additional to it. */
    int          cached_tokens;
    /* usage.completion_tokens_details.reasoning_tokens — thinking tokens. Already
       counted inside completion_tokens (billed as output); carried only so a
       reasoning model's hidden spend is visible rather than mysterious. */
    int          reasoning_tokens;
    double       tps;             /* predicted (generation) tokens/sec from timings (0 if absent) */
    double       prompt_tps;      /* prompt (prefill) tokens/sec from timings (0 if absent/cached) */
    /* Prefill WORK, as opposed to prompt size. The server evaluates only the tokens
     * that miss the KV cache, and prompt_tps is a rate over THOSE — so deriving a
     * duration as prompt_tokens/prompt_tps mixes a whole-prompt numerator with an
     * evaluated-only denominator. Measured on Qwen3.5-9B, a cache-hit turn reports
     * prompt_tokens=15442 against prompt_n=19: that derivation overstates prefill
     * time by ~813x, reproducibly. Use prompt_ms for time and prompt_n for work. */
    int          prompt_n;        /* tokens actually EVALUATED (excludes the cache hit) */
    double       prompt_ms;       /* server-measured wall-time to evaluate those tokens */
    double       predicted_ms;    /* server-measured wall-time to generate the completion */
    char        *finish_reason;   /* "stop" | "tool_calls" | "length" | ... (malloc'd, may be NULL) */
} SrvChatResult;

/* Stream a chat completion. messages_json and tools_json are caller-built JSON
 * ARRAYS (tools_json may be NULL/empty for no tools). samp: sampling knobs (NULL
 * => server defaults). n_predict<0 => unlimited. on_content / on_reasoning fire
 * per streamed delta for live display (either may be NULL). Returns a malloc'd
 * SrvChatResult (free with srvchat_free), or NULL on transport failure. */
SrvChatResult *srvchat_complete(int port, const char *messages_json, const char *tools_json,
                                const SrvSampling *samp, int n_predict,
                                void (*on_content)(const char *chunk, void *ud),
                                void (*on_reasoning)(const char *chunk, void *ud),
                                void *ud);

/* Best-of-N in ONE request: ask the server for n_choices independent
 * continuations of a single prefill. Measured on a 6720-token prompt with
 * Qwen3.5-9B: n=4 returns 4 answers in 11.5s (cached_tokens=6716) vs 33s for 4
 * separate concurrent requests, which each re-prefill the shared prefix — so 4
 * candidates cost ~1.95x one candidate, not 4x. Streaming is preserved: each SSE
 * chunk carries its choice index, and the callbacks receive it so the caller can
 * render one trajectory live (rendering all N interleaved is unreadable).
 *
 * Fills out[0..max_out) with malloc'd results (free each with srvchat_free) and
 * returns how many were written, or -1 on transport failure. */
int srvchat_complete_n(int port, const char *messages_json, const char *tools_json,
                       const SrvSampling *samp, int n_predict, int n_choices,
                       void (*on_content)(int choice, const char *chunk, void *ud),
                       void (*on_reasoning)(int choice, const char *chunk, void *ud),
                       void *ud, SrvChatResult **out, int max_out);

void srvchat_free(SrvChatResult *r);

/* ── Remote (hosted) OpenAI-compatible endpoint ──────────────────────────────
 * BASI's chat path is already plain /v1/chat/completions, so a hosted provider
 * (Fireworks, OpenAI, OpenRouter, Together, Groq, DeepSeek…) is the SAME client
 * pointed at a different base URL, with a bearer token and an explicit `model`.
 *
 * Once this is configured, every srvchat_complete/_n call goes to the remote and
 * the `port` argument is ignored — main.c spawns no llama-server at all. The
 * request body drops the llama-server-only extensions (cache_prompt,
 * repeat_penalty/repeat_last_n, min_p, reasoning_budget, chat_template_kwargs):
 * those are not in the OpenAI schema and hosted providers differ on whether an
 * unknown field is ignored or 400s. Add provider-specific knobs (e.g. Fireworks'
 * top_k) with BASI_API_EXTRA_JSON, which is merged into the body verbatim.
 *
 * base_url is the part BEFORE /chat/completions (e.g.
 * "https://api.fireworks.ai/inference/v1"); a trailing '/' is trimmed. api_key
 * may be NULL/empty for an unauthenticated endpoint. Returns 0 on success, -1 if
 * base_url or model is missing/unusable.
 *
 * The key is never placed on a command line (that would expose it in `ps` to
 * every user on the box) — it is passed to curl through a 0600 config file.
 *
 * NOTE: embeddings (srvchat_embed*) are unaffected and stay on the LOCAL
 * llama-server embedder — the remote is the chat model only. */
int srvchat_set_remote(const char *base_url, const char *api_key, const char *model);

/* Nonzero once srvchat_set_remote() has succeeded. */
int srvchat_remote_active(void);

/* The configured remote model id, or NULL when running locally. Never the key. */
const char *srvchat_remote_model(void);

/* The configured remote base URL, or NULL when running locally. */
const char *srvchat_remote_base(void);

/* Ask the provider for the configured model's context window, over the
 * OpenAI-standard GET <base>/models. Returns the length in tokens, or 0 when the
 * endpoint is missing, the model is not listed, or no length is reported — the
 * caller keeps its own default. One short request at startup, so the ctx meter
 * and the compaction trigger budget against the real window instead of a guess
 * (guessing LOW silently compacts what the provider would have accepted whole). */
int srvchat_remote_context_length(void);

/* POST one text to a llama-server /embedding endpoint (spawned with --embedding).
 * Writes up to max_dim floats of the pooled embedding into out and returns the
 * count written (the embedding dimension), or -1 on transport/parse failure. */
int srvchat_embed(int port, const char *text, float *out, int max_dim);

/* Embed `n` texts in ONE request ("content": [...]). Writes each embedding at
 * out[i * max_dim] (stride max_dim) and returns the dimension, or -1 on failure.
 *
 * Measured in-process on the Arc (Jina v5 small, ~200-char chunks), speedup vs
 * looping the single-text path: 1.6x @ N=16, 2.1x @ N=64, 2.2x @ N=256 — it
 * PLATEAUS around 2.2x by N~64, bottoming out at ~19 ms/text. The win is
 * per-REQUEST overhead, not model throughput, so it only appears on realistically
 * sized chunks; a micro-benchmark with 50-char strings on CPU shows no gain at
 * all (0.8x) and will mislead you. Do not expect more than ~2x from bigger batches.
 *
 * COST NOTE for anyone indexing tool output: ~19 ms/chunk at best means a 237 KB
 * page split at 200 chars (~1200 chunks) costs ~23 s to index. Chunk COARSE.
 *
 * Results are routed by the response's `index` field, not arrival order. */
int srvchat_embed_batch(int port, const char **texts, int n, float *out, int max_dim);

/* Self-test (BASI_SRV_CHAT_SELFTEST=1): spawn a server, send one message + a bash
 * tool, stream it, print structured content/reasoning/tool_calls/usage, tear down. */
void srvchat_selftest(const char *model_path, int ngl, int ctx);

#ifdef __cplusplus
}
#endif
#endif /* BASI_SRVCHAT_H */
