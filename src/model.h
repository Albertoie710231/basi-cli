#ifndef BASI_MODEL_H
#define BASI_MODEL_H

#include <stddef.h>
#include "llama.h"

/* One-shot init: registers the llama log callback + loads ggml backends. */
void model_init(void);

/* ── Launch config (returned by interactive picker) ────────────────── */
typedef struct {
    char *model_path;  /* malloc'd, caller frees */
    int   gpu_layers;
    int   ctx_size;
    float temperature;
} LaunchConfig;

LaunchConfig pick_model(void);

/* ── Chat template wrapper (handles non-Jinja fallback) ────────────── */
int apply_template(const char *tmpl,
                   const struct llama_chat_message *msgs, size_t n_msgs,
                   bool add_gen_prompt,
                   char *buf, size_t buf_size);

/* ── Generation ────────────────────────────────────────────────────── */
typedef struct {
    char  *text;          /* malloc'd, caller frees */
    size_t prompt_tokens;
    size_t gen_tokens;
    double prompt_time_s;
    double gen_time_s;
} GenerateResult;

GenerateResult generate(
    struct llama_context *ctx,
    const struct llama_vocab *vocab,
    struct llama_sampler *smpl,
    const char *prompt,
    size_t prompt_len);

/* Locate a "<tool>...</tool>" tag in the model's output text.
 * Returns a pointer into `text` to the byte after "<tool>", with *out_len
 * set to the body length. Returns NULL if no tag is found. */
const char *extract_tool_call(const char *text, size_t *out_len);

#endif /* BASI_MODEL_H */
