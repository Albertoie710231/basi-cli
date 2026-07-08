#ifndef BASI_SLASHMENU_H
#define BASI_SLASHMENU_H

/* Slash-command autocomplete dropdown for the line editor.
 *
 * Rendering strategy: the dropdown is drawn on the physical rows *below* the
 * input line using absolute cursor positioning. The caller supplies the input
 * row (obtained via a cursor-position report) and the bottom of the usable
 * region (one above the status bar); this module scrolls the region up only
 * when the menu would collide with that bottom edge. The input stays on a
 * single physical line, so the caller's relative-move editing is undisturbed.
 *
 * The pure parts (command table, active-detection, prefix filter) are trivially
 * testable; the terminal helpers (pushback-aware read, CPR) live here too so a
 * PTY harness can exercise the exact rendering code without the rest of BASI. */

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

typedef struct {
    const char *name;   /* includes the leading '/' */
    const char *desc;   /* one-line description shown in the dropdown */
    bool        takes_arg;
} SlashCmd;

/* The command table. Pass a non-NULL count to receive the entry count. */
const SlashCmd *slashmenu_table(int *count);

/* True when the dropdown should be active for the current buffer: the line
 * starts with '/' and no space precedes the cursor (still typing the command
 * word). Sets *prefix_len to the number of chars after '/' up to the cursor. */
bool slashmenu_active(const char *line, size_t len, size_t cursor,
                      size_t *prefix_len);

/* Fill idx[] (cap `max`) with table indices whose name (sans '/') begins with
 * the case-insensitive prefix. Empty prefix matches all. Returns match count
 * (capped at max). */
int slashmenu_filter(const char *prefix, size_t prefix_len, int *idx, int max);

/* Most rows shown at once; the visible window scrolls to keep the selection
 * in view when there are more matches than this. */
#define SLASHMENU_MAX_VISIBLE 8

/* Render state owned by the caller across keystrokes. Zero-initialize. */
typedef struct {
    bool open;   /* a menu is currently painted */
    int  r0;     /* input row the last paint anchored to (absolute) */
    int  rows;   /* number of menu rows last painted */
} SlashMenuState;

/* Paint the dropdown below the input line.
 *   r0            input row (1-based, from a cursor-position report)
 *   region_bottom last usable row (status-bar row minus one, or screen bottom)
 *   cols          terminal width
 *   idx/matchc    matches from slashmenu_filter; selected is highlighted
 *   promptw       visible prompt width; cursorcol logical cursor in the buffer
 * The cursor is returned to (input row, promptw+cursorcol) afterward. When the
 * paint scrolls the region, the anchor moves up; the caller learns the new row
 * via st->r0. */
void slashmenu_draw(FILE *out, SlashMenuState *st,
                    int r0, int region_bottom, int cols,
                    const int *idx, int matchc, int selected,
                    int promptw, int cursorcol);

/* Erase a painted dropdown and return the cursor to the input line. No-op when
 * st->open is false. */
void slashmenu_close(FILE *out, SlashMenuState *st, int promptw, int cursorcol);

/* ── Terminal helpers (shared with the editor + the PTY harness) ──────── */

/* Read one byte, draining an internal pushback buffer first (bytes captured
 * while parsing a cursor-position report land here so no keystroke is lost).
 * Returns 1 on a byte, 0/-1 like read(2). */
long slashmenu_read_byte(unsigned char *c);

/* Emit a cursor-position report request and parse the reply, returning the
 * 1-based row (or -1 on failure). Any non-report bytes read meanwhile are
 * pushed back for slashmenu_read_byte. */
int slashmenu_cpr_row(void);

/* Visible width of a prompt string, skipping CSI SGR escapes (…m). */
int slashmenu_visible_width(const char *s);

#endif /* BASI_SLASHMENU_H */
