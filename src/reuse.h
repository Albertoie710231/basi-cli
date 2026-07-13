#ifndef BASI_REUSE_H
#define BASI_REUSE_H

/* Reuse gate (Tier B of the "stop re-implementing code that already exists"
 * pillar). Deterministic guardrail: before an `edit` adds a NEW top-level
 * function, we embed it and compare against a project-wide symbol index. If a
 * near-duplicate already exists elsewhere in the tree, the edit is PAUSED and
 * the model is shown the match — it must reuse the existing function, or note
 * why a new one is needed and re-issue the edit (a re-issue always applies:
 * see the warn-once semantics in reuse.c).
 *
 * Opt-in: does nothing unless BASI_REUSE_GATE is set (1/on/true/yes). Degrades
 * to a no-op (never blocks an edit) when no embedding model is available.
 *
 * `path`   — the file being edited (matches against itself are ignored).
 * `replace`— the REPLACE text of one edit block (the code being added).
 * `search` — the SEARCH text of the same block (a function whose name appears
 *            here is being edited in place, not added — so it is skipped).
 *
 * Returns NULL to let the edit proceed, or a malloc'd model-facing message
 * (caller frees, and should return it as the tool result instead of applying
 * the edit) describing the near-duplicate(s). */
char *reuse_gate_check(const char *path, const char *replace, const char *search);

/* True if BASI_REUSE_GATE is enabled. Cheap; lets the edit path skip all reuse
 * work (extraction, matching) when the gate is off. */
int reuse_gate_enabled(void);

/* Deterministic reuse rewrite (output-side injection). If BASI_REUSE_AUTOFIX is
 * on and `replace` adds a new function that is a high-confidence, exact-arity
 * structural duplicate of an existing one, return a rewritten `replace` whose
 * body calls the existing function instead of re-implementing it. Returns NULL
 * if nothing was auto-fixed (caller then falls back to reuse_gate_check). The
 * caller owns the returned string and should apply the edit with it. */
char *reuse_gate_autofix(const char *path, const char *replace, const char *search);

/* Token clone similarity between two C/C++ function bodies (0..1), the
 * deterministic primitive the gate ranks candidates on. Exposed for evaluation
 * and for a future detect/report use. */
double reuse_similarity(const char *body_a, const char *body_b);

/* True if the behavior-preservation guard (BASI_REUSE_REGRESS) is enabled. */
int reuse_regress_enabled(void);

/* Behavior-preservation check on an edit. Given the file `path` and its OLD
 * (pre-edit) and NEW (post-edit) full contents, return a malloc'd model-facing
 * warning naming every EXISTING function whose behavior the edit CHANGED
 * (compiler-IR / parser-AST verified — a silent regression the type checker
 * misses, e.g. routing an existing shape's collision through new geometry), or
 * NULL if all existing behavior is preserved or the language has no verifier.
 * Warn-once per function name, so re-issuing the same edit applies it. Caller
 * frees the returned string and should return it as the tool result instead of
 * writing the file. */
char *reuse_regress_check(const char *path, const char *old_src, const char *new_src);

/* Free the in-process symbol store + warned-name set. Optional; process exit
 * reclaims everything. */
void reuse_shutdown(void);

#endif /* BASI_REUSE_H */
