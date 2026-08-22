#ifndef BASI_SAMPLING_H
#define BASI_SAMPLING_H
/* Per-model sampling defaults.
 *
 * BASI shipped ONE sampler chain for every model: temperature 0.4, no top_k, no
 * top_p, min_p 0.05, repeat_penalty 1.1. Those numbers came from BASI's own
 * native sampler and were reasonable for what was loaded at the time. They are
 * not reasonable as a universal default, and the failure is silent — running
 * Qwen3.8 that way contradicts every value on its model card, including the
 * direction of the one penalty BASI does set (the card asks for 1.0 in thinking
 * mode; BASI sent 1.1). Nothing printed, nothing warned, and the output just
 * quietly got worse.
 *
 * Two sources, deterministic, no network:
 *
 *   1. ~/.config/basi-cli/sampling — declared by the user, one line per model
 *      family. This is the only place the prose-only parameters can come from:
 *      generation_config.json carries temperature/top_k/top_p and NOTHING else,
 *      while min_p, the penalties, and the thinking-vs-instruct split live in
 *      the card's Best Practices text, which is not machine-readable.
 *
 *   2. generation_config.json sitting beside the weights — the vendor's own
 *      inference defaults, shipped in the repo. Covers the three it carries,
 *      for free, whenever the file is actually there.
 *
 * An explicit flag or environment variable always wins: this fills in what the
 * caller did not specify, it never overrides what they did. */
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Every field is "unset" as a negative number, so a profile can specify a subset
 * and leave the rest to whatever the caller already had. */
typedef struct {
    double temperature;     /* <0 = unset */
    double top_p;           /* <0 = unset */
    int    top_k;           /* <0 = unset */
    double min_p;           /* <0 = unset */
    double repeat_penalty;  /* <0 = unset */
    char   source[1216];   /* human-readable provenance, for the status line.
                             * Wide enough for a real config path plus the key:
                             * a truncated provenance line is worse than none,
                             * since the whole point is to say WHERE a value
                             * came from when it turns out to be wrong. */
    char   match[64];       /* which config key matched, "" if none */
} SamplingProfile;

/* Fill `out` with the recommended sampling for the model at `model_path`.
 * Returns the number of parameters found (0 = nothing applies, `out` all unset).
 * Never fails: an unreadable or absent config is simply "nothing applies". */
int sampling_profile_for(const char *model_path, SamplingProfile *out);

/* Path of the sampling config this build would read (for diagnostics/help). */
const char *sampling_config_path(void);

#ifdef __cplusplus
}
#endif
#endif /* BASI_SAMPLING_H */
