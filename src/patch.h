#ifndef BASI_PATCH_H
#define BASI_PATCH_H

#include <stdbool.h>
#include "util.h"   /* StringBuf */

/* edit tool — SEARCH/REPLACE blocks, applied with an exact->fuzzy cascade. */
char *execute_edit(const char *args);

/* Apply ONE SEARCH/REPLACE to `cur` in place: exact substring first, then a
 * whitespace-insensitive line match, re-indenting the replacement to the file's
 * real indentation. Both passes REFUSE an ambiguous SEARCH rather than silently
 * taking the first hit. Returns NULL on success, else a malloc'd model-facing
 * message explaining what to fix. `path`/`idx` only shape that message.
 *
 * Exposed because `factory` was matching with a bare strstr: no ambiguity check
 * (it patched the first of several identical sites), no whitespace tolerance. */
char *patch_replace_one(StringBuf *cur, const char *find, const char *repl,
                        const char *path, int idx);

/* True when a SEARCH/REPLACE pair cannot change the file, because the two texts
 * are identical. Such a block "applies" cleanly — SEARCH is found, the file is
 * rewritten byte-for-byte the same — so nothing downstream can tell it apart
 * from a real edit unless it is rejected here. Shared by the edit tool and by
 * `factory`, which measured four such theories as legitimate +0.00% results
 * before it reused this. */
bool patch_block_is_noop(const char *find, const char *repl);

#endif /* BASI_PATCH_H */
