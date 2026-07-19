#ifndef BASI_SYMBOLS_H
#define BASI_SYMBOLS_H
/* symbols tool — deterministic symbol enumeration for a source file, via ctags.
 *
 * Exists because the model cannot answer "what is defined in this file" with the
 * tools it had. Measured over 31 agentic runs: 24 clusters of 3+ consecutive greps,
 * all of them regex attempts at enumerating C function definitions, producing
 * 80/0/4/110/0/270/74/349 for a file that has exactly 72. Regexes cannot do this —
 * return types on their own line, multi-line signatures and macros defeat them —
 * so the fix is to remove the regex path rather than warn against it.
 *
 * Language-agnostic: whatever ctags parses (C/C++/Python/Go/JS/Rust/...), this
 * reports, so it is not a C-only affordance. */

/* List the symbols defined in `file`. `kind` optionally filters by ctags kind name
 * ("function", "struct", "macro", ...); NULL/empty means all kinds. Returns a
 * malloc'd report (caller frees), never NULL. */
char *execute_symbols(const char *file, const char *kind);

#endif /* BASI_SYMBOLS_H */
