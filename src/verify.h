#ifndef BASI_VERIFY_H
#define BASI_VERIFY_H

/* Tool entry: parse the Implementation Plan table from
 * .basi/plans/<current_plan_slug>.md and run each row's verify clause via
 * `bash -c`. Reports per-row OK / FAIL / SETUP (exit 127) / SKIP (empty).
 * Optional `args` is trimmed; if non-empty, only the row whose id matches
 * is run. Caller frees the returned string. */
char *execute_plan_verify(const char *args);

#endif /* BASI_VERIFY_H */
