#ifndef BASI_MODEL_H
#define BASI_MODEL_H

#include <stddef.h>
#include "basi_types.h"
#include "srvgen.h"   /* SrvSampling (server-backed generation knobs) */


/* ── Launch config (returned by interactive picker) ────────────────── */
typedef struct {
    char *model_path;  /* malloc'd, caller frees */
    int   gpu_layers;
    int   ctx_size;
    float temperature;
    int   spec_draft_mtp;  /* 1 = launch llama-server with --spec-type draft-mtp (MTP models) */
    int   flash_attn;      /* 1 = -fa on */
    int   cpu_moe;         /* 1 = --cpu-moe (MoE: experts to RAM, attention to GPU) */
    /* Name of the selected llama-server binary, or NULL when the BACKEND row was
     * hidden (fewer than two declared) or untouched. Borrowed from the backend
     * module's static list — not malloc'd, and outlives the caller. */
    const char *backend;
} LaunchConfig;

LaunchConfig pick_model(void);

/* Predicted VRAM in MB for `model_path` at ngl/ctx — the same figure the picker's
 * MEMORY row shows (GGUF tensor walk + per-layer KV + overhead term). Returns <0 if
 * the GGUF can't be read. Exposed so the launch path can hold the estimate up
 * against what the GPU actually reports afterwards. ngl<0 means "all layers". */
double basi_predict_vram_mb(const char *model_path, int ngl, int ctx);

/* Scan the model search dirs for .gguf files. Fills *out with a malloc'd array
 * of malloc'd path strings (caller frees each, then the array) and returns the
 * count. Used by the /model command to resolve a name substring to a path. */
int basi_list_models(char ***out);


/* ── Generation ────────────────────────────────────────────────────── */
typedef struct {
    char  *text;          /* malloc'd, caller frees */
    size_t prompt_tokens; /* whole prompt, incl. the KV-cache-hit prefix — this is
                             context OCCUPANCY, and not the work the server did */
    size_t gen_tokens;
    size_t prompt_n;      /* prompt tokens actually EVALUATED (cache misses only) */
    double prompt_time_s; /* real seconds spent prefilling, measured by the server.
                             Do NOT reconstruct it as prompt_tokens/prompt_tps: those
                             have different denominators and the result is ~800x high
                             on a cache hit. Pair it with prompt_n, never prompt_tokens. */
    double prompt_tps;    /* server's prefill rate, over prompt_n */
    double gen_time_s;
    size_t cached_tokens; /* subset of prompt_tokens served from the provider's prompt
                             cache (hosted APIs bill these at a discount). 0 locally. */
    size_t reasoning_tokens; /* thinking tokens, already inside gen_tokens. 0 locally. */
} GenerateResult;

/* Server-backed generation: when basi_srv_port>0, generate() delegates to a
   spawned llama-server (set by main.c). basi_srv_model = a vocab_only handle for
   deriving the tool grammar. */
extern int basi_srv_port;
/* Sampling knobs for the server /completion request (filled by main.c to mirror
   the native sampler chain). */
extern SrvSampling basi_srv_sampling;
/* When nonzero, generate_server() omits the tool-call grammar (deepsearch sets it
   around its own ReAct loop so the main tool grammar can't leak in). */
extern int basi_srv_suppress_grammar;
/* When nonzero, build_request() sets chat_template_kwargs.enable_thinking=false on
   every /v1/chat/completions request — the scoped, per-request form of
   BASI_NO_THINK. study_ground sets it around its grounding ReAct loop so the model
   navigates the code without a <think> block per tool call; the hypothesis step
   leaves it 0 and keeps reasoning. Caller-scoped, always restored (never setenv). */
extern int basi_srv_no_think;

/* Server-chat generation (item 6b): serialize `messages` (+ registered tools) and
   drive /v1/chat/completions. Returns the answer text (res.text) + the server's
   prompt token count (res.prompt_tokens) and fills tc_out/n_tc_out with the
   STRUCTURED tool calls (caller frees via basi_free_tool_calls). */
struct BasiToolCall;
GenerateResult generate_chat(const BasiMsg *messages, size_t msg_count,
                             struct BasiToolCall **tc_out, int *n_tc_out);

/* Locate a "<tool>...</tool>" tag in the model's output text.
 * Returns a pointer into `text` to the byte after "<tool>", with *out_len
 * set to the body length. Returns NULL if no tag is found. */
const char *extract_tool_call(const char *text, size_t *out_len);

#endif /* BASI_MODEL_H */
