#ifndef BASI_PLAN_H
#define BASI_PLAN_H

#include <stdbool.h>
#include <stddef.h>

#include "globals.h"

/* Returns NULL if the artifact is valid, otherwise a malloc'd, human-readable
 * description of all validation failures (one per line). */
char *validate_plan_artifact(const char *content, size_t content_len);

/* Tool entry: writes a Proposal-A3 plan artifact to .basi/plans/<slug>.md.
 * Validates first; refuses overwrite. Caller passes the body following
 * "<tool>plan_write " (i.e., the YAML frontmatter + body). */
char *execute_plan_write(const char *body);

/* Rewrites only the `status:` line in the frontmatter of
 * .basi/plans/<slug>.md so the file tracks runtime phase transitions
 * (Decision #5: file is source of truth). Returns 0 on success, -1 if the
 * file is missing or unreadable, -2 if the frontmatter has no status line. */
int rewrite_plan_status(const char *slug, const char *new_status);

/* Tool entry: declare unverified assumptions before drafting. Counts list
 * items in `body`; if ≥ 3 and we are in DRAFTING, transitions to SPIKE
 * (subject to SPIKE_MAX_CYCLES). Caller frees. */
char *execute_assumptions(const char *body);

/* Returns NULL if the spike artifact is valid, otherwise a malloc'd, human-
 * readable description of all validation failures (one per line). */
char *validate_spike_artifact(const char *content, size_t content_len);

/* Tool entry: writes (or appends, on subsequent cycles) a spike artifact to
 * .basi/plans/<slug>.spike.md. Validates first. Routes phase based on the
 * Decision line: PROCEED-TO-PLAN → DRAFTING; NEED-ANOTHER-SPIKE → stays SPIKE
 * unless the cycle cap is exhausted (then forced ABANDON); ABANDON → NONE. */
char *execute_spike_write(const char *body);

/* Per-phase tool gate. `command` is the raw <tool>...</tool> body (tool name
 * is the first whitespace-delimited token). Returns true if the tool is
 * allowed to run in `phase`. */
bool plan_tool_allowed(PlanPhase phase, const char *command);

/* Builds the "<tool> not allowed: <reason>" message for a blocked call.
 * Caller frees. Only valid when plan_tool_allowed returned false. */
char *plan_block_msg(PlanPhase phase, const char *command);

#endif /* BASI_PLAN_H */
