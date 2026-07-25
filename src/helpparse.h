#ifndef BASI_HELPPARSE_H
#define BASI_HELPPARSE_H
/* What flags does THIS llama-server accept?
 *
 * The launch script is composed from BASI's assumptions about llama-server's CLI,
 * but the binary is now selectable — and different builds can be of different
 * vintages. When a flag BASI emits isn't in that build, llama-server exits during
 * startup and the user gets "failed to start (see /tmp/basi_srvgen.log)". Parsing
 * `--help` lets the composer name the flag instead.
 *
 * `llama-server --help` turned out to be regular enough to parse: measured on the
 * Vulkan build, 648 lines / 4 sections / 246 flag lines, each beginning at column 0
 * with comma-separated flag aliases followed by an optional metavar
 * (`N`, `<0|1>`, `{a,b,c}`, `[on|off|auto]`, …) and then the description.
 *
 * This is a HEURISTIC over help text, so an unrecognized flag is reported as a
 * warning and never blocks a launch: a parser miss must not stop a valid command
 * from running. If the flag really is unsupported the server fails immediately
 * afterwards, and the warning explains why. */
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HelpSpec HelpSpec;

/* Parse `<server_bin> --help`. Cached by (path, size, mtime), so a rebuilt binary
 * is re-parsed but repeated launches are not. Returns NULL when the binary can't
 * be run — which is not an error here: the probe reports that case, and a backend
 * you can't load also can't print its help. */
const HelpSpec *helpspec_for(const char *server_bin);

/* Is `flag` (with dashes, e.g. "--spec-type" or "-ngl") accepted? */
int helpspec_has_flag(const HelpSpec *hs, const char *flag);

/* How many distinct flag aliases were parsed — for diagnostics and tests. */
int helpspec_flag_count(const HelpSpec *hs);

/* Check every dash-prefixed token on the exec line(s) of a launch script against
 * `server_bin`'s help. Writes up to `max_out` unknown flags into `out` (each
 * malloc'd; caller frees) and returns how many were found. Returns -1 if the help
 * could not be parsed at all, so the caller can stay quiet instead of guessing. */
int helpspec_check_script(const char *script_path, const char *server_bin,
                          char **out, int max_out);

#ifdef __cplusplus
}
#endif
#endif /* BASI_HELPPARSE_H */
