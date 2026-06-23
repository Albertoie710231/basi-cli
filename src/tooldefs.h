#ifndef BASI_TOOLDEFS_H
#define BASI_TOOLDEFS_H

#include "chat_tmpl.h"   /* BasiToolDef */

/* The static native-tool table (flat JSON schemas). Sets *n. */
const BasiToolDef *basi_tool_defs(int *n);

/* Translate a parsed native tool call into the command string that
 * execute_tool() already understands (so all dispatch + plan-phase gating is
 * reused). Returns a malloc'd string (caller frees), or NULL for an unknown
 * tool name. */
char *basi_build_command(const char *name, const char *arguments_json);

#endif /* BASI_TOOLDEFS_H */
