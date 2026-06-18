#ifndef BASI_CHAT_TMPL_H
#define BASI_CHAT_TMPL_H

#include <stddef.h>
#include <stdbool.h>
#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Render `msgs` into a prompt using the MODEL'S OWN chat template, via
   llama.cpp's jinja engine (libllama-common). This is what makes BASI drive
   each model in its native format (Gemma, DeepSeek, custom merges, …) instead
   of a one-size-fits-all ChatML fallback. Returns a malloc'd string (caller
   frees), or NULL on any failure — callers fall back to the legacy path. */
char *basi_render_chat(const struct llama_model *model,
                       const struct llama_chat_message *msgs, size_t n_msgs,
                       bool add_gen_prompt);

#ifdef __cplusplus
}
#endif

#endif /* BASI_CHAT_TMPL_H */
