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
    /* --cpu-moe: pin MoE expert tensors to system RAM while every attention layer
     * goes to the GPU. On an A3B this is both faster and smaller than fitting
     * whole layers (33.97 vs 24.60 tok/s, ~4.0 vs ~5.5 GB), because only a couple
     * of experts fire per token but an offloaded layer must carry all of them. */
    int         cpu_moe;          /* 1 = emit --cpu-moe */
    int         flash_attn;       /* -fa on */
    int         jinja;            /* --jinja (needed for the chat/tools endpoints) */
    /* Server slots. >1 emits `-np N --kv-unified`, which best-of-N REQUIRES: the
     * server caps the OpenAI `n` parameter at the slot count (with -np 1 it 400s
     * "Value must be between 1 <= value <= 1"). --kv-unified is not optional
     * either — without it `-c` is DIVIDED across slots, so `-c 8192 -np 4` leaves
     * 2048 per slot and silently rejects any longer prompt. */
    int         n_parallel;       /* 0/1 = omit (single slot, no best-of-N) */
    const char *reasoning_format; /* --reasoning-format (e.g. "auto"); NULL = omit */
    /* The selected llama-server binary's name and its per-binary default flags
     * (e.g. "-b 2048 -ub 2048" for SYCL, where ubatch is the prefill lever).
     * backend_name is recorded in the script as a "# BASI-BACKEND:" marker so a
     * backend switch invalidates script reuse — without it, switching binaries on
     * an unchanged model would silently re-run the old script. NULL = omit. */
    const char *backend_name;
    const char *extra_flags;
} SrvLaunch;

/* Write a standalone, runnable llama-server launch script to `path` (creating the
 * parent dir). The script carries a "# BASI-MODEL: <path>" marker so BASI can tell
 * a stale/other-model script from a user-edited one for the same model. Returns 0
 * on success, -1 on error. */
int srvgen_write_launch_script(const SrvLaunch *cfg, const char *path);

/* If `path` exists and its "# BASI-MODEL:" marker equals model_path, return 1
 * (reuse the user's script as-is); else 0 (missing / stale / different model). */
int srvgen_script_matches(const char *path, const char *model_path);

/* Same, for the "# BASI-BACKEND:" marker. Reuse must be gated on model AND
 * backend: the model marker alone would let a backend switch reuse a script that
 * still names the old binary, so the switch would appear to do nothing.
 * A script with no backend marker (hand-written, or predating this) counts as a
 * match for the default backend, so existing scripts keep being honored. */
int srvgen_script_backend_matches(const char *path, const char *backend_name);

/* 1 if `path`'s exec line references `server_bin`. Lets a reused script that was
 * hand-edited to a different binary be reported rather than silently obeyed. */
int srvgen_script_uses_bin(const char *path, const char *server_bin);

/* Read the -ngl / -c the script will ACTUALLY run with (each left untouched if
 * absent). Returns 1 if the script was readable.
 *
 * Needed because script reuse is keyed on model+backend, so a reused or hand-edited
 * script keeps its own -ngl/-c and silently overrides the ones BASI was invoked
 * with. Anything reasoning about memory must use these, not the requested values,
 * or it describes a server that isn't running. */
int srvgen_script_params(const char *path, int *ngl, int *ctx);

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
