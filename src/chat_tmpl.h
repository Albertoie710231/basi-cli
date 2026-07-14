#ifndef BASI_CHAT_TMPL_H
#define BASI_CHAT_TMPL_H

#include <stddef.h>
#include <stdbool.h>
#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Native tool-calling (phase 2a) ────────────────────────────────────
   BASI presents tools to the model in its OWN trained format (Qwen/Hermes
   <tool_call> JSON, Llama-3, Mistral, …) by handing definitions to
   llama.cpp's common_chat machinery, which also parses the calls back. */

/* One tool definition. `parameters` is a flat JSON Schema string. */
typedef struct {
    const char *name;
    const char *description;
    const char *parameters;   /* JSON Schema (object); "{}" for no args */
} BasiToolDef;

/* One parsed tool call returned by basi_parse_tool_calls (caller-owned). */
typedef struct {
    char *name;        /* malloc'd */
    char *arguments;   /* malloc'd JSON object string */
} BasiToolCall;

/* Register the tool set (once, at startup). Definitions are referenced, not
   copied, so they must outlive use (static literals are fine). */
void basi_set_tools(const BasiToolDef *defs, int n);

/* Number of tools currently advertised to the model. A self-contained
   sub-generation can clear the schemas and restore the prior state via this. */
int  basi_tools_registered(void);

/* 1 if tools are registered AND this model's template supports tool calls
   (i.e. common_chat resolves a non-content-only format); else 0 → caller
   uses the legacy <tool> prose path. */
int  basi_tools_active(const struct llama_model *model);

/* Phase 2b — build the GBNF grammar sampler that constrains decoding to a valid
   tool call in this model's format (the grammar common_chat already derives from
   the tool set). Lazy: inactive until a tool-call trigger, so free text/thinking
   still works. The caller adds the returned sampler to its chain BEFORE the final
   selector and MUST llama_sampler_reset it between generations. NULL if this
   format has no tool grammar, or on any error (caller proceeds unconstrained). */
struct llama_sampler *basi_tool_grammar_sampler(const struct llama_model *model);

/* Server-backend: the same tool-call grammar serialized as spliceable
   llama-server request fields ("grammar":…,"grammar_lazy":…,"grammar_triggers":…),
   no outer braces. NULL if no grammar. Caller frees. */
char *basi_tool_grammar_json(const struct llama_model *model);

/* Parse a finished assistant generation into tool calls. Returns the count
   (0 = none / parse failure → treat as a plain answer). On >0, *out points to
   a malloc'd array the caller frees with basi_free_tool_calls. */
int  basi_parse_tool_calls(const char *text, BasiToolCall **out);
void basi_free_tool_calls(BasiToolCall *calls, int n);

/* The reasoning ("thinking") open/close delimiters this model's template
   declares, as derived by common_chat (e.g. "<think>"/"</think>", or Gemma's
   "<|channel>thought"/"<channel|>"). Returns 1 and points start and end at
   interned strings when the model supports thinking with non-empty tags; else
   0, and the caller keeps its own default. Pointers are valid until the next
   render — copy them if you need to hold them. */
int  basi_thinking_tags(const char **start, const char **end);

/* Render `msgs` into a prompt using the MODEL'S OWN chat template, via
   llama.cpp's jinja engine (libllama-common). This is what makes BASI drive
   each model in its native format (Gemma, DeepSeek, custom merges, …) instead
   of a one-size-fits-all ChatML fallback. When tools are registered (see
   basi_set_tools) they are advertised to the model here. Returns a malloc'd
   string (caller frees), or NULL on any failure — callers fall back. */
char *basi_render_chat(const struct llama_model *model,
                       const struct llama_chat_message *msgs, size_t n_msgs,
                       bool add_gen_prompt);

#ifdef __cplusplus
}
#endif

#endif /* BASI_CHAT_TMPL_H */
