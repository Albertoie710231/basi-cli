#ifndef BASI_TOOLSTAT_H
#define BASI_TOOLSTAT_H

#include <stdio.h>
#include <stdbool.h>

/* Every tool hands back a plain string, so a tool whose backend is DOWN and a
 * tool that ran fine and found nothing come back looking identical. The model
 * reads both as "nothing there", quietly works around the outage, and writes a
 * confident answer; the user sees no sign that a capability was missing for the
 * whole run. That is how a dead web_search went unnoticed here for months — the
 * runs did not fail, they just silently got worse.
 *
 * This module is the missing distinction. Tools report infrastructural failure
 * — backend unreachable, binary missing, no credentials — as something other
 * than an empty result, and the run says out loud what was broken instead of
 * leaving it to be inferred from a tool-call log. */

/* Marker on the string handed to the model, so the distinction survives into the
 * transcript and not just into our counters. */
#define BASI_TOOL_ERROR "TOOL ERROR: "

/* Count one dispatch of `tool`. Called centrally, for every tool, every call —
 * the denominator in "4/4 calls failed". */
void basi_toolstat_call(const char *tool);

/* Record that the call in flight failed for an infrastructural reason, NOT
 * because it legitimately found nothing. The first reason seen per tool is kept
 * for the report; later ones are counted but not stored. */
void basi_toolstat_fail(const char *tool, const char *reason);

/* Record the failure and return the malloc'd BASI_TOOL_ERROR string to hand
 * back, for the common case where a tool returns its error straight to the
 * caller. Caller frees. */
char *basi_toolstat_failed(const char *tool, const char *reason);

/* True if anything failed this run. */
bool basi_toolstat_any_failed(void);

/* End-of-run summary — one line per tool that failed, with how many of its calls
 * failed and why. Prints nothing when everything worked, so a healthy run stays
 * quiet. */
void basi_toolstat_report(FILE *f);

/* Row i of the running counts (cumulative for the run), for callers that keep
 * their own record — local telemetry diffs these per turn. False past the end. */
bool basi_toolstat_row(int i, const char **name, int *calls, int *fails);

#endif /* BASI_TOOLSTAT_H */
