#ifndef BASI_GLOBALS_H
#define BASI_GLOBALS_H

#include <stdbool.h>
#include <signal.h>

/* Model / runtime sizing constants. */
#define MAX_TOKENS    8192
#define CONTEXT_SIZE  32768

/* CLI-set flags. */
extern bool debug_mode;

/* Approval-cache flags toggled by the y/n/a prompt. */
extern bool bash_always_allowed;
extern bool apply_patch_always_allowed;
extern bool scaffold_always_allowed;

/* Permission mode (set via /permissions slash command). */
typedef enum {
    PERM_DEFAULT,       /* prompt for bash, apply_patch, scaffold */
    PERM_ACCEPT_EDITS,  /* auto-approve apply_patch + scaffold; bash still prompts */
    PERM_BYPASS         /* auto-approve everything */
} PermissionMode;
extern PermissionMode permission_mode;
const char *perm_mode_name(PermissionMode m);

/* Plan-mode phase machine (Decision #5 in .basi/plans/knowledge-base.md).
 * Replaces the legacy bool flag. PHASE_NONE means no plan in progress. */
typedef enum {
    PHASE_NONE,
    PHASE_DRAFTING,
    PHASE_SPIKE,
    PHASE_PREMORTEM,
    PHASE_ACTIVE
} PlanPhase;
extern PlanPhase plan_phase;
extern char *current_plan_slug;     /* malloc'd; NULL when phase == PHASE_NONE */
const char *plan_phase_name(PlanPhase p);
const char *plan_phase_banner(PlanPhase p);   /* NULL when no banner needed */

/* Spike-phase budget (Decision #5; mock-session edge #5).
 * `spike_cycles` is per-plan: capped at SPIKE_MAX_CYCLES; the next would force
 * ABANDON. `spike_calls` is per-cycle: tool-calls other than spike_write while
 * phase == SPIKE. Both reset on /plan <slug> entry and /plan off. */
#define SPIKE_MAX_CYCLES   3
#define SPIKE_MAX_CALLS   12
#define SPIKE_TOKEN_HINT 8000     /* advisory only, surfaced in banner */
extern int spike_cycles;
extern int spike_calls;

/* Sigint / Ctrl+T globals — set in handlers, read everywhere generate() runs. */
extern volatile sig_atomic_t generation_interrupted;
extern volatile sig_atomic_t show_thinking;

/* When set, generate() suppresses all terminal output (token stream, thinking
 * box) but still builds and returns its text. Used by deepsearch to run many
 * internal rounds silently while it prints its own progress lines. */
extern volatile sig_atomic_t generate_quiet;

/* When set, generate() KEEPS <think>...</think> in the returned text instead of
 * stripping it. Used by deepsearch so a reasoning model's chat history stays
 * consistent (its prior turns include their thinking) and it can think freely. */
extern volatile sig_atomic_t generate_keep_think;

/* y/n/a approval prompt for risky tools.
 * Returns: 0 = deny, 1 = allow once, 2 = always (caller sets flag). */
int request_approval(const char *tool_label, const char *cmd);

/* Read-before-edit tracker. A file is "seen" once it has been viewed this
 * session via read/head/tail/grep/cat/readfile; apply_patch uses this to
 * refuse editing a file the model has not grounded itself in. Paths are
 * canonicalised, so relative/absolute spellings of the same file match. */
void read_tracker_mark(const char *path);
bool read_tracker_seen(const char *path);

#endif /* BASI_GLOBALS_H */
