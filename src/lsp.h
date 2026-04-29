#ifndef BASI_LSP_H
#define BASI_LSP_H

/* code_context tool — clangd-backed structural query.
 * Argument string: "<file> <symbol>" (whitespace-separated, no colons).
 * Returns malloc'd output (caller frees). */
char *execute_code_context(const char *args);

/* Tear down the cached clangd subprocess (if any). Call on exit. */
void lsp_shutdown(void);

#endif /* BASI_LSP_H */
