#ifndef BASI_BACKEND_H
#define BASI_BACKEND_H
/* Selectable llama-server binaries.
 *
 * The /model picker is a terminal GUI for composing a llama-server command line:
 * every row is one flag, and the result is written to .basi/run-llama-server.sh
 * and exec'd. One field was missing from that composer — the binary itself, which
 * was hardcoded to the Vulkan build in six places. This module supplies it.
 *
 * The candidates are DECLARED BY THE USER, not discovered: BASI has no business
 * guessing which llama.cpp builds on the box are meant to be used. With no config
 * file the built-in Vulkan default applies, so behavior is unchanged.
 *
 * Why the choice must be persisted rather than hand-edited into the script: a
 * hand-edited launch script IS honored today (srvgen_script_matches only checks
 * the model marker), but a /model switch invalidates that marker and the script
 * is overwritten — silently reverting to Vulkan. Only a declared, saved selection
 * survives regeneration. */
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Plenty: this is a list of local builds, not a registry. */
#define BACKEND_MAX 8

typedef struct {
    char name[32];         /* menu label, e.g. "vulkan" / "sycl-aot" */
    char server_bin[512];  /* absolute path to llama-server (argv[0]) */
    char extra_flags[256]; /* per-binary default flags, e.g. "-b 2048 -ub 2048" */
} Backend;

/* The compatibility default: Vulkan runs on any NVIDIA/AMD/Intel GPU, so it is
 * the fallback whenever nothing else is declared or resolvable. Overridable at
 * build time (-DBASI_DEFAULT_SERVER_BIN=...) for packaging. */
#ifndef BASI_DEFAULT_SERVER_BIN
#define BASI_DEFAULT_SERVER_BIN "/home/alberto/llama.cpp/build_vulkan/bin/llama-server"
#endif

/* Parse the config (idempotent; later calls are no-ops). Always succeeds: a
 * missing, empty or malformed file leaves the built-in Vulkan entry in place.
 * Returns the number of usable entries. */
int backend_load(void);

/* The declared entries, for the picker's BACKEND row. */
const Backend *backend_list(int *n);

/* Look up by name; NULL if not declared. */
const Backend *backend_get(const char *name);

/* The backend to launch with. Precedence, highest first:
 *   1. $BASI_SERVER_BIN  — the pre-existing override; kept working verbatim so
 *                          existing scripts and the factory harness don't break
 *   2. $BASI_BACKEND     — name from the declared list
 *   3. backend_select()  — the picker's choice / the saved `backend=` line
 *   4. "vulkan", else the first declared entry
 * Never returns NULL. */
const Backend *backend_active(void);

/* Record the selection (from the picker or the saved config). Unknown names are
 * ignored, so a stale `backend=` in the config can't break a launch. Returns 1 if
 * the name resolved, 0 otherwise. */
int backend_select(const char *name);

/* The EXPLICIT selection only (picker or saved config), or "" if none — unlike
 * backend_active(), which always resolves to something. Kept separate so an
 * implicit fallback or a one-off $BASI_SERVER_BIN is never persisted as if the
 * user had chosen it. */
const char *backend_selected_name(void);

/* Does this binary actually run? Executes `<server_bin> --version` and checks the
 * exit status — a real load test, so it catches a missing runtime env (SYCL needs
 * oneAPI on LD_LIBRARY_PATH), a wrong-arch binary and a half-built tree alike.
 * Returns 0 if it runs; else -1, with a human-readable reason in `err` (the
 * loader's own message when there is one). */
int backend_probe(const Backend *b, char *err, size_t errn);

/* Best-effort remediation for a failed probe, derived from the loader's message:
 * a binary missing the Intel runtime libs needs the oneAPI env, which BASI
 * deliberately does NOT source for you (the launch script stays a plain exec, and
 * the environment is inherited). Returns "" when nothing specific applies. */
const char *backend_probe_hint(const Backend *b, const char *err);

/* Path of the config file (XDG-aware), for diagnostics. */
void backend_config_path(char *out, size_t n);

#ifdef __cplusplus
}
#endif
#endif /* BASI_BACKEND_H */
