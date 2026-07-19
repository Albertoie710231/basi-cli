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

/* The llama-server launch configuration the model menu edits and BASI runs from.
 * Since generation is delegated to llama-server, "how to run the model" IS this
 * command — BASI writes it to a standalone, editable shell script and execs it. */
typedef struct {
    const char *server_bin;       /* llama-server binary */
    const char *model_path;
    int         ngl;              /* -ngl */
    int         ctx;              /* -c   */
    const char *host;             /* --host (default 127.0.0.1) */
    int         port;             /* --port (BASI connects here) */
    const char *spec_type;        /* --spec-type (e.g. "draft-mtp"); NULL/"" = off */
    int         spec_nmax;        /* --spec-draft-n-max */
    int         flash_attn;       /* -fa on */
    int         jinja;            /* --jinja (needed for the chat/tools endpoints) */
    /* Server slots. >1 emits `-np N --kv-unified`, which best-of-N REQUIRES: the
     * server caps the OpenAI `n` parameter at the slot count (with -np 1 it 400s
     * "Value must be between 1 <= value <= 1"). --kv-unified is not optional
     * either — without it `-c` is DIVIDED across slots, so `-c 8192 -np 4` leaves
     * 2048 per slot and silently rejects any longer prompt. */
    int         n_parallel;       /* 0/1 = omit (single slot, no best-of-N) */
    const char *reasoning_format; /* --reasoning-format (e.g. "auto"); NULL = omit */
} SrvLaunch;

/* Write a standalone, runnable llama-server launch script to `path` (creating the
 * parent dir). The script carries a "# BASI-MODEL: <path>" marker so BASI can tell
 * a stale/other-model script from a user-edited one for the same model. Returns 0
 * on success, -1 on error. */
int srvgen_write_launch_script(const SrvLaunch *cfg, const char *path);

/* If `path` exists and its "# BASI-MODEL:" marker equals model_path, return 1
 * (reuse the user's script as-is); else 0 (missing / stale / different model). */
int srvgen_script_matches(const char *path, const char *model_path);

/* 1 if `path` launches with at least `need` slots AND --kv-unified — i.e. it can
 * serve best-of-`need`. Used to warn instead of letting every turn 400 when a
 * reused/hand-edited script predates best-of-N. */
int srvgen_script_has_slots(const char *path, int need);

/* Spawn llama-server by exec'ing a launch script (`bash path`), then poll /health
 * on `port`. Same return contract as srvgen_spawn. */
pid_t srvgen_spawn_script(const char *path, int port, const char *logpath, int timeout_s);

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
