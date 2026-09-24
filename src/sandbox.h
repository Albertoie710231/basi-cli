#ifndef BASI_SANDBOX_H
#define BASI_SANDBOX_H

#include <stdbool.h>

/* Re-execute BASI inside a bubblewrap sandbox: writes only to the project, /tmp,
 * ~/.cache and BASI's own config/data (+ .basi/sandbox and --allow entries);
 * reads everywhere except hidden secrets (~/.ssh, ~/.claude, …). Returns only if
 * already sandboxed, turned off (--no-sandbox / BASI_SANDBOX=0), or bwrap is
 * unavailable — each case says so on stderr. See sandbox.c. */
void sandbox_maybe_reexec(int argc, char **argv);

bool sandbox_disabled_by_args(int argc, char **argv);

#endif /* BASI_SANDBOX_H */
