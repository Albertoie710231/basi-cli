#ifndef BASI_MD_H
#define BASI_MD_H

#include <stddef.h>

/* Streaming markdown → ANSI renderer for the live answer stream.
 *
 * Fed the assistant's answer text token-by-token as it generates; emits
 * ANSI-styled terminal output. Paragraph text streams live (token feel
 * preserved); block constructs (headings, lists, blockquotes, rules, fenced
 * code) are recognised at line start; inline spans (**bold**, *italic*,
 * `code`, [text](url)) are styled as they complete. Display-only: it never
 * changes the text the model produced (that is captured separately by the
 * caller). All state is module-global and reset by md_begin(). */

void md_begin(void);              /* reset state for a fresh answer */
void md_feed(const char *s, size_t n);  /* stream answer bytes */
void md_end(void);                /* flush any pending line / open spans */

#endif /* BASI_MD_H */
