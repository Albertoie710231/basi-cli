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

/* Per-phase tool gate. `command` is the raw <tool>...</tool> body (tool name
 * is the first whitespace-delimited token). Returns true if the tool is
 * allowed to run in `phase`. */
bool plan_tool_allowed(PlanPhase phase, const char *command);

/* Builds the "<tool> not allowed: <reason>" message for a blocked call.
 * Caller frees. Only valid when plan_tool_allowed returned false. */
char *plan_block_msg(PlanPhase phase, const char *command);

#endif /* BASI_PLAN_H */
