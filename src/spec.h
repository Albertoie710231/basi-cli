#ifndef BASI_SPEC_H
#define BASI_SPEC_H
/* Speculative-decoding shim (C interface over llama.cpp's C++ common_speculative /
 * common_sampler). Milestone 1: MTP driver + a self-test path; not yet wired into
 * generate(). Default-off — nothing here runs unless explicitly enabled. */
#include "llama.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct basi_spec basi_spec;

/* Set up speculative decoding over ctx_tgt. `spec_type` e.g. "draft-mtp".
 * model_path / n_ctx / n_gpu_layers must mirror how ctx_tgt was created (used to
 * build the matching MTP draft context). Returns NULL if the type is
 * none/unknown or setup fails — the caller then decodes normally. Greedy when
 * temp<=0 (the lossless setting). */
basi_spec *basi_spec_init(struct llama_model *model,
                          struct llama_context *ctx_tgt,
                          const char *model_path,
                          int n_ctx, int n_gpu_layers,
                          const char *spec_type, int n_max,
                          unsigned int seed, float temp);

/* One committed token; is_eog != 0 for the end-of-generation token. */
typedef void (*basi_spec_emit)(llama_token tok, int is_eog, void *ud);

/* Run one generation: `prompt` is the already-tokenized prompt (n_prompt tokens,
 * incl. any BOS). Speculates until EOG, context-full, or `cap` tokens (cap<=0 =
 * unbounded). Calls emit() per committed token. Returns tokens generated, -1 on
 * error. */
int basi_spec_run(basi_spec *s,
                  const llama_token *prompt, int n_prompt,
                  long cap, basi_spec_emit emit, void *ud);

/* Cumulative draft stats since init (for the bench / status line). */
void basi_spec_stats(const basi_spec *s, long *drafted, long *accepted);

void basi_spec_free(basi_spec *s);

/* Hidden A/B: plain-greedy vs spec-greedy on a fixed prompt (env: BASI_SPEC type,
 * BASI_SPEC_NMAX, BASI_SPEC_N, BASI_SPEC_PROMPT). Reports lossless + speedup.
 * Triggered by BASI_SPEC_SELFTEST=1 in main() right after the context is built. */
void basi_spec_selftest(struct llama_model *model, struct llama_context *ctx,
                        const char *model_path, int n_ctx, int n_gpu_layers);

#ifdef __cplusplus
}
#endif
#endif /* BASI_SPEC_H */
