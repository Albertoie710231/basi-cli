#ifndef BASI_VRAMOBS_H
#define BASI_VRAMOBS_H
/* Observed VRAM: closing the loop between the picker's estimate and the GPU.
 *
 * `estimate_memory` predicts VRAM from the GGUF (exact per-layer weights, per-layer
 * KV with SWA windows) plus a fixed overhead term. The weight and KV terms are
 * exact; the overhead term is a calibration, and MEASURED it over-predicts by
 * ~1.2 GB on a dense 7B at ctx 32768 (predicted 6.6 GB, actual 5.3) because it
 * charges 600 MB + ctx*n_embd*6 where the real compute buffer was 205 MiB.
 *
 * That error is in the safe direction — it under-fits layers rather than OOMing —
 * and its slack is load-bearing: `-ub 2048` costs +518 MiB of compute buffer that
 * the estimator does not model at all. So rather than retune the constant blind,
 * BASI records what each model ACTUALLY used and corrects with that.
 *
 * The estimate stays the fallback: a model never launched has no observation, and
 * the first launch of any config is what produces one. */
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Exact observation for this (model, ngl, ctx). 1 if found, else 0. */
int vramobs_get(const char *model_path, int ngl, int ctx,
                double *predicted_mb, double *observed_mb);

/* Correction (observed - predicted) learned for this model at ANY recorded config,
 * most recent first. 1 if found, else 0. Used when the current ngl/ctx has never
 * been launched: the error is dominated by the fixed overhead term, so carrying the
 * offset across configs is closer than carrying nothing. */
int vramobs_offset(const char *model_path, double *offset_mb);

/* Record what a launch actually used. Replaces any entry for the same
 * (model, ngl, ctx). Best-effort: any I/O failure is silently ignored, since
 * losing an observation must never fail a launch. */
void vramobs_record(const char *model_path, int ngl, int ctx,
                    double predicted_mb, double observed_mb);

/* Apply the best available correction to `predicted_mb` for this config: an exact
 * observation wins, else the model's offset, else the input unchanged. Sets
 * *was_measured to 1 when an exact observation was used. */
double vramobs_correct(const char *model_path, int ngl, int ctx,
                       double predicted_mb, int *was_measured);

#ifdef __cplusplus
}
#endif
#endif /* BASI_VRAMOBS_H */
