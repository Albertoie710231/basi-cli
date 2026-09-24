#ifndef BASI_TELEMETRY_H
#define BASI_TELEMETRY_H

#include <stdbool.h>
#include <stddef.h>

/* LOCAL telemetry. One JSON line per agent turn, appended to
 *   ~/.local/share/basi-cli/telemetry.jsonl
 * and NOTHING ELSE: no socket is opened, nothing is uploaded, and it is not
 * collected for training anyone's model. It exists so the user can see how BASI
 * actually behaves on their machine — which tools fail, which model runs out of
 * rounds, which tool is never touched — and decide what to change.
 *
 * What a record holds: time, project directory, backend + model, outcome
 * (answered / capped / repeat_stopped / parse_failed / interrupted), rounds,
 * tokens, speed, and per-tool call/failure COUNTS. What it never holds: the
 * prompt, the answer, tool arguments or tool output.
 *
 * Off switch, strongest first:
 *   BASI_TELEMETRY=0|off        this run only
 *   basi-cli telemetry off      persisted in ~/.config/basi-cli/telemetry
 * The first run that records anything says so once, with the off command. */

bool telemetry_enabled(void);

/* Who is answering, set once the backend is resolved. `backend` is "local" or
 * the remote base URL; `mode` is "repl" or "oneshot". Copied. */
void telemetry_set_context(const char *backend, const char *model, const char *mode);

/* Bracket one agent turn. end writes the record (no-op when disabled). */
void telemetry_turn_begin(void);
void telemetry_turn_end(const char *outcome, int rounds, int elision_resets,
                        size_t prompt_tokens, size_t gen_tokens, double gen_tps);

/* `basi-cli telemetry [status|on|off|show [N]|purge|path]`. Returns exit code. */
int telemetry_cmd(int argc, char **argv);

#endif /* BASI_TELEMETRY_H */
