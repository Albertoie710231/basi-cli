#ifndef BASI_TOOLDEFS_H
#define BASI_TOOLDEFS_H

#include "chat_tmpl.h"   /* BasiToolDef */

#ifdef __cplusplus
extern "C" {
#endif

/* The active tool table: the static native tools, followed by any extras
 * registered below. Sets *n.
 *
 * Every place that saves and restores the advertised tool set (deepsearch, the
 * compaction summary, a study's grounding phase) goes through this function, so
 * making it the merge point is what keeps dynamically discovered tools alive
 * across those round trips without touching a single one of those call sites. */
const BasiToolDef *basi_tool_defs(int *n);

/* Append a caller-owned block of tool defs to the table (MCP servers use this).
 * Definitions are REFERENCED, not copied, so they must outlive the registration;
 * pass NULL/0 to drop them again. Replaces any previous block. */
void basi_tooldefs_set_extra(const BasiToolDef *defs, int n);

/* Translate a parsed native tool call into the command string that
 * execute_tool() already understands (so all dispatch + plan-phase gating is
 * reused). Returns a malloc'd string (caller frees), or NULL for an unknown
 * tool name. */
char *basi_build_command(const char *name, const char *arguments_json);

#ifdef __cplusplus
}
#endif

#endif /* BASI_TOOLDEFS_H */
