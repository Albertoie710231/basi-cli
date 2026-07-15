#ifndef BASI_SESSION_H
#define BASI_SESSION_H

#include <stdio.h>
#include <stddef.h>

#include "basi_types.h"

/* Returns malloc'd path: ~/.local/share/basi-cli/projects/<encoded-cwd>/
 * with the directory created. NULL on failure. */
char *session_dir_path(void);

/* Append one role/content pair to the open session log. */
void session_write_record(FILE *fp, const char *role, const char *content);

/* Interactive picker: list previous session files in `dir`, return malloc'd
 * path to the chosen one, or NULL on cancel. */
char *session_picker(const char *dir);

/* Resume from a saved session log: append user/assistant turns into the
 * given llama chat-message buffer (up to a 30%-of-context char budget). */
void session_load_into(const char *path,
                       BasiMsg **messages,
                       size_t *msg_count,
                       size_t *msg_cap,
                       int n_ctx);

/* Open a fresh session log in `dir`. Optionally returns the malloc'd path. */
FILE *session_open_new(const char *dir, char **out_path);

#endif /* BASI_SESSION_H */
