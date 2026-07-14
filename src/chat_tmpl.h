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
typedef struct BasiToolCall {
    char *name;        /* malloc'd */
    char *arguments;   /* malloc'd JSON object string */
} BasiToolCall;

/* Register the tool set (once, at startup). Definitions are referenced, not
   copied, so they must outlive use (static literals are fine). */
void basi_set_tools(const BasiToolDef *defs, int n);

/* Number of tools currently advertised to the model. A self-contained
   sub-generation can clear the schemas and restore the prior state via this. */
int  basi_tools_registered(void);

/* Serialize the registered tools / a message array into the OpenAI JSON the
   server's /v1/chat/completions endpoint expects (item 6, pure-HTTP path).
   Both return a malloc'd string (caller frees); basi_tools_to_json returns NULL
   when no tools are registered. */
char *basi_tools_to_json(void);
char *basi_messages_to_json(const struct llama_chat_message *msgs, int n_msgs);

/* Free a tool-call array (name/arguments strdup'd by the caller side). */
void basi_free_tool_calls(BasiToolCall *calls, int n);

#ifdef __cplusplus
}
#endif

#endif /* BASI_CHAT_TMPL_H */
