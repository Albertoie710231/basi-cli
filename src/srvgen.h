#ifndef BASI_SRVGEN_H
#define BASI_SRVGEN_H
/* Server-backed generation spike (M1): drive a spawned llama-server over its
 * /completion SSE stream instead of linking libllama in-process. Proves the
 * pivot (Pi-style architecture) and gets MTP spec-decode "for free" (the server
 * drives it correctly, which the native path could not). Default-off; nothing
 * here runs unless BASI_SERVER_SELFTEST is set. */
#include <sys/types.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Spawn llama-server for model_path at ngl/ctx on 127.0.0.1:port. `extra` is a
 * space-separated string of extra flags (e.g. "-fa on --spec-type draft-mtp
 * --spec-draft-n-max 1") or NULL. Server output goes to logpath. Blocks until
 * /health returns OK or timeout_s elapses. Returns the child pid, or -1. */
pid_t srvgen_spawn(const char *server_bin, const char *model_path, int ngl, int ctx,
                   const char *extra, int port, const char *logpath, int timeout_s);

/* Sampling knobs for a /completion request, mirroring BASI's native sampler
 * chain (penalties → min_p → temp → dist). A field set to its "omit" sentinel is
 * left out of the request so the server uses its own default. NULL SrvSampling*
 * => server defaults everywhere (greedy off). */
typedef struct {
    double temperature;    /* <0 → omit; 0 → greedy (argmax) */
    double repeat_penalty; /* <=1 → omit */
    int    repeat_last_n;  /* <0 → omit (window for repeat_penalty) */
    double min_p;          /* <0 → omit */
    int    top_k;          /* <=0 → omit */
    double top_p;          /* <0 or >=1 → omit */
    long   seed;           /* <0 → omit (random each request) */
} SrvSampling;

/* Stream one /completion. Calls emit(content_chunk, ud) per SSE chunk as tokens
 * arrive. Returns the full generated text (malloc'd; caller frees) or NULL.
 * Fills *tps (tok/s) and *n_out (tokens) if non-NULL.
 * samp: sampling knobs (NULL => server defaults).
 * grammar_fragment: caller-provided valid JSON spliced verbatim into the request
 *   (e.g. basi_tool_grammar_json()'s output), or NULL. */
char *srvgen_complete(int port, const char *prompt, int n_predict,
                      const SrvSampling *samp, const char *grammar_fragment,
                      void (*emit)(const char *chunk, void *ud), void *ud,
                      double *tps, int *n_out);

void srvgen_kill(pid_t pid);

/* M1 self-test: spawn a server (+ spec flags from BASI_SPEC/BASI_SPEC_NMAX),
 * stream a fixed prompt live to stdout, report tok/s, tear down. */
void srvgen_selftest(const char *model_path, int ngl, int ctx);

#ifdef __cplusplus
}
#endif
#endif /* BASI_SRVGEN_H */
