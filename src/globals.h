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

/* Sigint / Ctrl+T globals — set in handlers, read everywhere generate() runs. */
extern volatile sig_atomic_t generation_interrupted;
extern volatile sig_atomic_t show_thinking;

/* y/n/a approval prompt for risky tools.
 * Returns: 0 = deny, 1 = allow once, 2 = always (caller sets flag). */
int request_approval(const char *tool_label, const char *cmd);

#endif /* BASI_GLOBALS_H */
