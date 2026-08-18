#ifndef BASI_MCP_H
#define BASI_MCP_H
/* Model Context Protocol client — BASI as an MCP HOST.
 *
 * An MCP server is a process (stdio) or an HTTP endpoint that advertises tools
 * over JSON-RPC 2.0. This module connects to the servers the user declared,
 * asks each for its tool list, and hands those tools to the SAME registry the
 * native tools live in (basi_tooldefs_set_extra), so they are advertised to the
 * model, constrained by the server-side tool grammar, scoped by `--tools`, and
 * saved/restored by every existing path — with no per-call-site special casing.
 *
 * DUAL-ERA. The protocol split in two at revision 2026-07-28:
 *   - MODERN (2026-07-28+): no handshake. Every request carries its protocol
 *     version, client info and client capabilities in `params._meta` under
 *     reserved `io.modelcontextprotocol/` keys, and `server/discover` reports
 *     what a server supports. Sessions are gone; a connection is not a
 *     conversation.
 *   - LEGACY (2025-11-25 and earlier): an `initialize` request negotiates a
 *     version once, followed by a `notifications/initialized`, and the session
 *     is the connection.
 * The ecosystem holds both, so this client speaks both and detects which per
 * server, exactly as the spec's backward-compatibility rules prescribe: probe
 * with `server/discover`; a DiscoverResult or a recognized modern error
 * (UnsupportedProtocolVersion, -32022) means modern, and ANY other error — or
 * silence — means legacy. The fallback is deliberately NOT keyed to one error
 * code: legacy servers answer an unknown pre-initialize method with whatever
 * their SDK picked (-32601 and -32602 are both common) or never answer at all.
 *
 * Everything here is protocol and transport. POLICY — whether a call is allowed
 * to run, and what the user is shown before it does — stays in main.c with the
 * rest of the approval gate, so an MCP tool is governed by the same y/n/a prompt
 * and the same permission modes as bash. */

#include "chat_tmpl.h"   /* BasiToolDef */

#ifdef __cplusplus
extern "C" {
#endif

/* Read the config, connect every enabled server, and list its tools. Returns the
 * MCP tool defs (owned here, alive until mcp_shutdown) and sets *n; NULL/0 when
 * MCP is disabled, unconfigured, or no server yielded a usable tool. A server
 * that fails to start is reported and skipped — one broken entry must not stop
 * BASI from launching.
 *
 * `extra_config` is an additional config file (--mcp-config), applied last.
 * `verbose` prints a per-server status line as each one connects. */
const BasiToolDef *mcp_init(const char *extra_config, int verbose, int *n);

/* Was any server configured at all? Distinguishes "no MCP servers" from
 * "servers configured but all of them failed", which need different advice. */
int mcp_configured(void);

/* Is `name` one of ours? Names are mangled `mcp__<server>__<tool>` so tools from
 * different servers cannot collide (the spec scopes tool-name uniqueness to a
 * single server, and two servers each exposing `search` is expected). */
int mcp_is_tool(const char *name);

/* The server's own readOnlyHint annotation: 1 read-only, 0 not, -1 unstated.
 * SELF-REPORTED and therefore untrusted — the spec is explicit that annotations
 * from an untrusted server mean nothing. Used to LABEL the approval prompt, never
 * to skip it. */
int mcp_tool_readonly(const char *name);

/* The server label a tool came from (static, alive until shutdown), or NULL. */
const char *mcp_tool_server(const char *name);

/* Invoke a tool. `arguments_json` is the model's already-parsed JSON object.
 * Returns malloc'd text for the model (caller frees) — never NULL for a known
 * tool: a transport failure, a JSON-RPC error and a tool-execution error all come
 * back as readable text, because an agent loop recovers from a message and cannot
 * recover from a NULL. */
char *mcp_call_tool(const char *name, const char *arguments_json);

/* Human-readable status (malloc'd, caller frees) for `/mcp`.
 * detail 0 = one line per server; 1 = also every tool it exposes. */
char *mcp_status_report(int detail, const char *server_filter);

/* Tear down one server and connect it again, re-registering the merged tool
 * table. `server_filter` NULL = all. Returns a malloc'd report. */
char *mcp_reconnect(const char *server_filter);

/* Close stdin on every child, wait briefly, then SIGTERM/SIGKILL — the spec's
 * shutdown sequence, in that order. Idempotent. */
void mcp_shutdown(void);

#ifdef __cplusplus
}
#endif
#endif /* BASI_MCP_H */
