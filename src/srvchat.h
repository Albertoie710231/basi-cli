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
    int          prompt_tokens;   /* from usage (0 if absent) */
    int          completion_tokens;
    double       tps;             /* predicted tokens/sec from timings (0 if absent) */
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

void srvchat_free(SrvChatResult *r);

/* POST one text to a llama-server /embedding endpoint (spawned with --embedding).
 * Writes up to max_dim floats of the pooled embedding into out and returns the
 * count written (the embedding dimension), or -1 on transport/parse failure. */
int srvchat_embed(int port, const char *text, float *out, int max_dim);

/* Self-test (BASI_SRV_CHAT_SELFTEST=1): spawn a server, send one message + a bash
 * tool, stream it, print structured content/reasoning/tool_calls/usage, tear down. */
void srvchat_selftest(const char *model_path, int ngl, int ctx);

#ifdef __cplusplus
}
#endif
#endif /* BASI_SRVCHAT_H */
